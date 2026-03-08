# Genos-m 长上下文可行性执行手册

## 1. 已知前提

- 原始数据：约 `13,000` 个 FASTA 文件
- 数据类型：约 `13,000` 个细菌基因组
- 平均长度：约 `2.5M bp / genome`
- 主干模型：Genos-m，`1M` max context
- 主干训练 dtype：`bf16`
- 当前仓库的 readout 路线：冻结 hidden states -> JEPA readout -> sequence embedding

这份手册的目标是把数据准备、脚本执行、排期和资源预算统一成一个可执行版本。


## 2. 这批 FASTA 应该怎么准备

### 2.1 输入文件

先准备一个文本文件，例如 `fasta_files.txt`，格式如下：

```text
/abs/path/genome_00001.fasta
/abs/path/genome_00002.fna
/abs/path/genome_00003.fa
...
```

要求：

- 一行一个 FASTA 路径
- 路径必须是绝对路径
- 文件名最好稳定，不要后续再改名

### 2.2 样本切分原则

对这类长基因组数据，必须按 **genome 级别** 切分 train/val/test，不能按 window 随机切，否则会发生严重泄漏。

推荐：

- train: `80%`
- val: `10%`
- test: `10%`

以 `13,000` 个 genome 计，约等于：

- train: `10,400`
- val: `1,300`
- test: `1,300`

### 2.3 多 contig FASTA 的处理方式

为了让 `128k` 和 `1M` bucket 能稳定构造，推荐默认策略是：

- 把同一 FASTA 内所有 contig 按原顺序拼接
- contig 之间插入 `256` 个 `N`
- 所有非 `ACGTN` 字符统一替换成 `N`

这样做的原因是：

- 对碎片化装配更稳
- 更容易拿到 `128k` 和 `1M` 的窗口
- 不会因为跨 contig 硬拼接而完全丢掉边界信息，`N` 间隔会显式标记边界

如果后续你只想对完整染色体做 `1M` pilot，可以再在 FASTA 列表层面单独筛一批高质量 assembly。

### 2.4 推荐长度桶和样本数

这次目标是“验证可行性”，不是先做最大规模数据生产，所以样本数不要一开始就铺满。

推荐长度桶：

- `8k`
- `32k`
- `128k`
- `1M` pilot

推荐样本数：

| Bucket | 长度 | Train | Val | Test | 总样本数 |
|---|---:|---:|---:|---:|---:|
| 8k | 8,192 | 3,000 | 400 | 400 | 3,800 |
| 32k | 32,768 | 1,500 | 200 | 200 | 1,900 |
| 128k | 131,072 | 300 | 50 | 50 | 400 |
| 1M pilot | 1,048,576 | 40 | 10 | 10 | 60 |

这样设计的原因：

- `8k` / `32k` 足够做 baseline 和 head 对比
- `128k` 足够验证长上下文扩展是否成立
- `1M` 先做 pilot，验证技术可行性和资源边界

### 2.5 标签数据怎么准备

JEPA readout 训练本身可以不带标签，但如果你要跑 linear probe，就需要额外准备一个标签文件。

推荐格式：`TSV` 或 `CSV`，至少包含两列：

```text
genome_id	label
GCF_xxx	3
GCF_yyy	1
...
```

要求：

- `genome_id` 要和脚本生成的 genome_id 对齐
- 当前 probe 脚本只支持单标签
- 分类标签用整数
- 回归标签用浮点数

如果暂时没有标签，可以先只跑无监督 JEPA + embedding export，把 linear probe 留到后面。


## 3. 推荐目录结构

跑完脚本后，建议目录长这样：

```text
outputs/
  feasibility_dataset/
    manifests/
      8k/
        train.jsonl
        val.jsonl
        test.jsonl
      32k/
      128k/
      1m/
    metadata/
      genome_metadata.json
      bucket_summary.json
    hidden_states/
      8k/
        train/
        val/
        test/
      32k/
      128k/
      1m/
  feasibility_runs/
    8k_mean/
    8k_attention/
    8k_qformer/
    32k_mean/
    32k_qformer/
    128k_qformer/
  feasibility_eval/
    8k_qformer/
    ...
```


## 4. 磁盘预算

现在仓库已经补上了 `bf16` hidden-state 存储和加载链路。对于这批数据，建议统一使用：

- extraction model dtype: `bf16`
- hidden-state save dtype: `bf16`
- training/eval load dtype: `bf16`

### 4.1 单样本 hidden-state 大小

按 `hidden_size=1024` 计算：

| Bucket | 长度 | `bf16` 单样本 | `float32` 单样本 |
|---|---:|---:|---:|
| 8k | 8,192 | 16 MiB | 32 MiB |
| 32k | 32,768 | 64 MiB | 128 MiB |
| 128k | 131,072 | 256 MiB | 512 MiB |
| 1M | 1,048,576 | 2 GiB | 4 GiB |

### 4.2 按推荐样本数估算

| Bucket | 总样本数 | `bf16` 估算 | `float32` 估算 |
|---|---:|---:|---:|
| 8k | 3,800 | 59.4 GiB | 118.8 GiB |
| 32k | 1,900 | 118.8 GiB | 237.5 GiB |
| 128k | 400 | 100.0 GiB | 200.0 GiB |
| 1M pilot | 60 | 120.0 GiB | 240.0 GiB |
| 合计 | 6,160 | 398.2 GiB | 796.3 GiB |

实际落盘时再加上：

- `attention_mask`
- `.pt` 序列化开销
- manifest / metadata

所以建议预留：

- `bf16` 方案：`430-500 GiB`
- `float32` 方案：`850-950 GiB`

结论：

- 这批数据必须优先用 `bf16`
- `1M` pilot 即便用 `bf16`，单样本仍约 `2 GiB`


## 5. 直接可运行的脚本

已经落到仓库中的脚本如下：

- [scripts/01_prepare_feasibility_data.sh](../scripts/01_prepare_feasibility_data.sh)
- [scripts/02_extract_hidden_states.sh](../scripts/02_extract_hidden_states.sh)
- [scripts/03_train_feasibility_suite.sh](../scripts/03_train_feasibility_suite.sh)
- [scripts/04_export_and_probe.sh](../scripts/04_export_and_probe.sh)
- [scripts/05_run_feasibility_suite.sh](../scripts/05_run_feasibility_suite.sh)

辅助脚本：

- [scripts/prepare_fasta_windows.py](../scripts/prepare_fasta_windows.py)

### 5.1 第一步：生成长度桶 manifest

```bash
cd /Users/linyuxiang/Project/ContextAgg

export FASTA_LIST=/abs/path/to/fasta_files.txt
export OUTPUT_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset

bash scripts/01_prepare_feasibility_data.sh
```

如果有标签文件：

```bash
export LABELS_TSV=/abs/path/to/genome_labels.tsv
bash scripts/01_prepare_feasibility_data.sh
```

输出：

- `manifests/<bucket>/<split>.jsonl`
- `metadata/genome_metadata.json`
- `metadata/bucket_summary.json`

### 5.2 第二步：提取 hidden states

```bash
cd /Users/linyuxiang/Project/ContextAgg

export MODEL_PATH=/Users/linyuxiang/Project/ContextAgg/model/Genos_m.GQA-MoE32-2-5B-8k
export DATASET_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset
export HS_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset/hidden_states
export GPU_IDS="0 1 2 3 4 5 6 7"
export MODEL_DTYPE=bfloat16
export SAVE_DTYPE=bfloat16

bash scripts/02_extract_hidden_states.sh
```

说明：

- 脚本会按 bucket/split 自动循环
- 默认使用 JSONL manifest
- 会跳过已经存在的 `.pt` 文件

### 5.3 第三步：训练 readout 基线

```bash
cd /Users/linyuxiang/Project/ContextAgg

export HS_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset/hidden_states
export RUN_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_runs
export GPU_IDS="0 1 2 3 4 5"

bash scripts/03_train_feasibility_suite.sh
```

默认会训练：

- `8k_mean`
- `8k_attention`
- `8k_qformer`
- `32k_mean`
- `32k_qformer`
- `128k_qformer`

如果要把 `1M` pilot 训练也打开：

```bash
export RUN_1M_PILOT=1
bash scripts/03_train_feasibility_suite.sh
```

### 5.4 第四步：导出 embedding 并跑 linear probe

```bash
cd /Users/linyuxiang/Project/ContextAgg

export HS_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset/hidden_states
export RUN_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_runs
export EVAL_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_eval
export GPU_IDS="0 1 2 3"

bash scripts/04_export_and_probe.sh
```

如果 hidden-state 文件里没有标签，脚本会自动跳过 linear probe，只保留 embedding export。

### 5.5 一键跑完整流程

```bash
cd /Users/linyuxiang/Project/ContextAgg

export FASTA_LIST=/abs/path/to/fasta_files.txt
export MODEL_PATH=/Users/linyuxiang/Project/ContextAgg/model/Genos_m.GQA-MoE32-2-5B-8k
export OUTPUT_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset
export GPU_IDS="0 1 2 3 4 5 6 7"

bash scripts/05_run_feasibility_suite.sh
```

### 5.6 云端多卡并发调度

当前仓库的核心代码依然是**单进程、单 GPU** 路径，因此这里的 `torchrun` / `deepspeed` 用法要理解准确：

- 可以把它们当作**单卡任务启动器**
- 不要把当前代码直接当成 DDP/FSDP 训练脚本
- 当前脚本只支持“多个单卡任务并发”，不支持“一个实验吃满多卡”

已经支持的环境变量：

- `GPU_IDS`
- `LAUNCHER`
- `MASTER_PORT_BASE`
- `LOG_ROOT`

其中：

- `LAUNCHER=python`：直接用 `python`
- `LAUNCHER=torchrun`：每个任务用 `torchrun --nproc_per_node=1`
- `LAUNCHER=deepspeed`：每个任务用 `deepspeed --num_gpus=1`

也就是说：

- `torchrun` / `deepspeed` 在这里是**进程管理工具**
- 不是多卡数据并行

#### 推荐用法一：直接用 `python` 并发

这是最稳的默认方案。

```bash
cd /Users/linyuxiang/Project/ContextAgg

export GPU_IDS="0 1 2 3 4 5 6 7"
export LAUNCHER=python
export LOG_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/cloud_logs/train

bash scripts/03_train_feasibility_suite.sh
```

#### 推荐用法二：云端统一用 `torchrun`

如果你们的云端规范统一要求 `torchrun`，就这样用：

```bash
cd /Users/linyuxiang/Project/ContextAgg

export GPU_IDS="0 1 2 3 4 5 6 7"
export LAUNCHER=torchrun
export MASTER_PORT_BASE=29600
export LOG_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/cloud_logs/extract

bash scripts/02_extract_hidden_states.sh
```

这时每个任务实际等价于：

```bash
CUDA_VISIBLE_DEVICES=<gpu> torchrun --standalone --nnodes=1 --nproc_per_node=1 ...
```

#### 推荐用法三：云端统一用 `deepspeed`

如果你们机器上 `deepspeed` 是默认启动器，就这样用：

```bash
cd /Users/linyuxiang/Project/ContextAgg

export GPU_IDS="0 1 2 3 4 5 6 7"
export LAUNCHER=deepspeed
export MASTER_PORT_BASE=29700
export LOG_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/cloud_logs/eval

bash scripts/04_export_and_probe.sh
```

这时每个任务实际等价于：

```bash
CUDA_VISIBLE_DEVICES=<gpu> deepspeed --master_port <port> --num_gpus 1 ...
```

#### 后台长跑建议

在云端服务器上，建议用 `nohup` 或 `tmux`。

`nohup` 示例：

```bash
cd /Users/linyuxiang/Project/ContextAgg

export FASTA_LIST=/abs/path/to/fasta_files.txt
export MODEL_PATH=/Users/linyuxiang/Project/ContextAgg/model/Genos_m.GQA-MoE32-2-5B-8k
export OUTPUT_ROOT=/Users/linyuxiang/Project/ContextAgg/outputs/feasibility_dataset
export GPU_IDS="0 1 2 3 4 5 6 7"
export LAUNCHER=torchrun
export MASTER_PORT_BASE=29600

nohup bash scripts/05_run_feasibility_suite.sh \
  > /Users/linyuxiang/Project/ContextAgg/outputs/nohup_feasibility.log 2>&1 &
echo $!
```

查看日志：

```bash
tail -f /Users/linyuxiang/Project/ContextAgg/outputs/nohup_feasibility.log
tail -f /Users/linyuxiang/Project/ContextAgg/outputs/cloud_logs/train/128k_qformer.log
```

如果你更习惯 `tmux`，推荐按阶段开 3 个窗口：

- 窗口 1：manifest + extraction
- 窗口 2：training
- 窗口 3：export + probe + 监控

#### 端口规划

并发使用 `torchrun` / `deepspeed` 时，必须避免 `master_port` 冲突。

推荐：

- extraction：`MASTER_PORT_BASE=29600`
- training：`MASTER_PORT_BASE=29700`
- eval：`MASTER_PORT_BASE=29800`

脚本内部会按 `JOB_INDEX` 自动递增端口。

#### 占卡策略

脚本现在是**按波次调度**：

- 一轮最多同时跑 `GPU_IDS` 中 GPU 数量个任务
- 同一波结束后，再启动下一波
- 不会在 GPU 数小于任务数时把多个训练任务堆到同一张卡上

这对于当前单卡代码路径是正确的默认行为。

#### 不要这样用

当前仓库代码还没有做 DDP/FSDP/ZeRO 的真正接入，所以不要这样跑：

```bash
torchrun --nproc_per_node=8 train.py ...
deepspeed --num_gpus 8 train.py ...
```

这样不会得到正确的多卡训练收益，反而大概率会出现：

- 多进程重复读相同数据
- 重复保存 checkpoint
- 日志混乱
- 输出目录互相覆盖

如果后面你要把单个 `128k` 或 `1M` 实验真正改成多卡训练，那是下一步代码改造，不是当前这套 shell 调度能解决的。


## 6. 周计划

### Week 1：数据清点与 8k/32k 准备

目标：

- 完成 FASTA 目录核对、manifest 生成、8k/32k hidden-state 提取

动作：

- 跑 `01_prepare_feasibility_data.sh`
- 抽样检查 `bucket_summary.json`
- 先提 8k/32k hidden states
- 用少量样本做一次 smoke test

交付物：

- 可用的 `8k/32k` manifests
- 第一批 hidden-state 文件
- smoke test 日志

### Week 2：8k/32k baseline

目标：

- 先确认 readout 路线是通的

动作：

- 跑 `8k_mean / 8k_attention / 8k_qformer`
- 跑 `32k_mean / 32k_qformer`
- 导出 embedding
- 如果有标签，跑第一版 linear probe

交付物：

- baseline loss 曲线
- 第一版 probe 指标
- 是否保留 `qformer` 作为主线的结论

### Week 3：128k 扩展

目标：

- 验证长上下文扩展能力

动作：

- 提 128k hidden states
- 跑 `128k_qformer`
- 做 chunk size / query 数量的小规模消融

交付物：

- `128k` 是否稳定
- 与 `32k` 对比的质量变化
- 是否继续推进 `1M` pilot 的决定

### Week 4：1M pilot

目标：

- 验证 `1M` 在工程上和表征上是否可行

动作：

- 只对 pilot 集提 `1M` hidden states
- 先跑 export 和 probe
- 稳定后再开 `1M` JEPA 短程训练

交付物：

- `1M` pilot 运行日志
- 显存、磁盘、时长实测
- 最终 go / no-go 建议


## 7. 预估耗时 / 显存 / 磁盘表

下面是基于 `A800 80G`、`bf16` 存储、当前仓库实现的粗估，适合排期，不适合作为 SLA。

### 7.1 数据准备与 hidden-state 提取

| 阶段 | 数据规模 | GPU | 预估耗时 | 单卡显存 | 产出磁盘 |
|---|---|---:|---:|---:|---:|
| manifest 生成 | 13k FASTA 全量扫描 | CPU | 2-6 小时 | - | < 20 GiB |
| 8k 提取 | 3,800 样本 | 3 卡并行 | 6-12 小时 | 18-24 GiB | 59-65 GiB |
| 32k 提取 | 1,900 样本 | 3 卡并行 | 8-16 小时 | 28-40 GiB | 118-125 GiB |
| 128k 提取 | 400 样本 | 2 卡并行 | 10-20 小时 | 45-60 GiB | 100-105 GiB |
| 1M pilot 提取 | 60 样本 | 1 卡 | 24-48 小时 | 70-78 GiB | 120-125 GiB |

### 7.2 Readout 训练与评测

| 阶段 | 任务 | GPU | 预估耗时 | 单卡显存 | 备注 |
|---|---|---:|---:|---:|---|
| Smoke | tiny_test + 1 epoch | 1 卡 | 0.5-1 小时 | < 10 GiB | 先验证链路 |
| Phase 1 | 8k baselines | 3 卡并行 | 2-4 小时 / run | 6-10 GiB | mean / attention / qformer |
| Phase 1 | 32k baselines | 2 卡并行 | 4-8 小时 / run | 8-14 GiB | mean / qformer |
| Phase 2 | 128k qformer | 1-2 卡 | 6-12 小时 | 12-20 GiB | 可附带小规模消融 |
| Phase 3 | 1M export-only pilot | 1 卡 | 2-6 小时 | 8-16 GiB | 先不训练 |
| Phase 3 | 1M qformer 3 epochs | 1 卡 | 8-16 小时 | 18-28 GiB | 仅在 export 稳定后开启 |

### 7.3 总体排期建议

在 `A800 80G x 8` 全可用、I/O 正常的情况下，推荐总排期：

- 最快：`3 周`
- 稳妥：`4 周`

更现实的关键路径通常是：

1. `1M` hidden-state 提取
2. 标签对齐与 probe 数据准备
3. `128k` / `1M` 的磁盘与 I/O 吞吐


## 8. 这版执行方案的关键结论

- 对这 13k 个细菌基因组，不要一开始就全量提 `1M` hidden states
- 先用 `8k/32k/128k` 建立 readout 可行性，再做 `1M` pilot
- hidden states 必须优先用 `bf16` 存，否则磁盘和 I/O 成本会迅速失控
- 当前仓库更适合“8 卡并行跑多个实验”，而不是“1 个 8 卡分布式训练”
- 真正的高风险项仍然是 `1M` extraction，而不是 readout head 本身
