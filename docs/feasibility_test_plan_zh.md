# ContextAgg 可行性测试方案

## 1. 范围

这个仓库本身**并不**验证完整大模型的语言建模能力，而是验证一个更具体的问题：

> 能否将 Genos-m 主干网络产生的冻结 hidden states，通过一个 readout JEPA 头压缩成有用的序列级 embedding？

结合当前代码，仓库实际支持的是：

- 从本地 HF checkpoint 提取 hidden states：`data/extract_hidden_states.py`
- 在冻结 hidden states 上训练 JEPA readout：`train.py`
- 导出 embedding：`eval/export_embeddings.py`
- 线性探针评测：`eval/linear_probe.py`

因此，这个项目真正要验证的可行性是：

1. 主干 hidden states 本身已经包含可用的长上下文信息
2. readout head 能把这些信息恢复到 `z_seq` 中
3. 当序列长度从 `8k -> 32k -> 128k -> 1M` 增长时，这种方法仍然稳定


## 2. 当前仓库的边界

在制定测试方案前，需要先明确当前实现的限制：

- `train.py` 是单进程、单 GPU，没有 DDP/FSDP
- `data/extract_hidden_states.py` 也是单进程、单 GPU
- `configs/v1.yaml` 默认 `data.max_length=131072`，不是 `1M`
- hidden state 提取当前保存的是 `last_hidden.float().cpu()`，也就是 `float32`
- `HiddenStateDataset` 加载后也会转成 `float32`
- 仓库目前只有 linear probe，没有 retrieval benchmark，也没有长上下文 serving benchmark

这意味着：

- 在 `A800 80G x 8` 环境下，最直接的用法是**并行跑多个实验**，而不是把 8 卡用于单个分布式训练任务
- 如果不先优化 hidden-state 的存储和提取方式，那么完整 `1M` 端到端验证更适合作为**pilot**，而不是大规模基准测试


## 3. 为什么这个项目仍然适合做可行性验证

仓库里的模型配置声明了：

- `hidden_size=1024`
- `num_local_experts=32`
- `num_experts_per_tok=2`
- `max_position_embeddings=1048576`

这与背景信息是一致的：

- 总参数 4.7B 的 MoE
- 32 个专家、每个 token 激活 2 个专家
- 激活参数约 330M
- 使用 GQA
- 最大上下文 1M

这个 readout 路线在长上下文下具有计算上的可行性，原因是：

- `models/chunk_pooling.py` 已经实现了 chunk pooling
- 默认 `64/64` 的 chunk pooling 会把 `1,000,000` token 压到约 `15,625` 个 pooled token
- 因此 Q-Former 的 cross-attention 复杂度是 `num_queries x pooled_length`，而不是 `raw_length x raw_length`

所以更可能的瓶颈是 **hidden-state 的提取与存储**，而不是 JEPA readout 本身。


## 4. 成功标准

只有在下面 4 个门槛都通过时，才建议认定“模型方案可行”。

### Gate A：工程可行性

- extraction、training、export、probe 全流程运行时没有 NaN / inf / shape 错误
- checkpoint 能正常产出
- 在验证集上可以正常导出 embedding

### Gate B：表征有效性

- `qformer` 相比简单 readout baseline（如 `mean` 或 `last`）在至少 2 个下游任务上更好
- linear probe 性能明显高于 majority-class 或 random baseline
- JEPA 训练 loss 有下降趋势，embedding 没有塌缩

### Gate C：长度扩展性

- 从 `8k -> 32k -> 128k` 的质量退化可控
- 同一类 head 在 `128k` 下仍然成立
- 在一个小规模 held-out 集合上，`1M` pilot 能完成 export 和下游评测

### Gate D：资源可行性

- `128k` 训练能在单张 `A800 80G` 上稳定运行
- `1M` inference / export 至少能在 `batch size = 1` 下放进显存
- pilot 级别的磁盘开销和运行时间在可接受范围内


## 5. 数据设计

由于仓库没有内置评测数据集，建议把测试数据分成两层。

### 第一层：合成控制任务

这一层是必须的，因为它能隔离并检验 embedding 是否真的保留了长距离信息。

- motif presence 分类
- motif count 回归
- motif order 分类：模式 A 在 B 前面，还是 B 在 A 前面
- long-range interaction 分类：两个 motif 相隔很远但共同决定标签
- distractor robustness：关键信号不变，但无关背景逐步增多

建议：

- 每个任务都做长度分桶：`8k`、`32k`、`128k`
- `1M` 先只做一个小规模 pilot 子集
- 标签生成过程尽量确定性，这样失败时更容易定位问题

### 第二层：真实下游任务

至少选择 3 个真实任务：

- 2 个分类任务
- 1 个回归任务

如果这个项目面向生物序列，比较自然的任务包括：

- taxonomy 或 family 分类
- phenotype / host / attribute 分类
- 连续分值回归

每个样本都可以按当前仓库支持的格式保存：

```python
{
  "hidden_states": Tensor[L, D],
  "attention_mask": Tensor[L],
  "label": Tensor[...] or scalar
}
```


## 6. 存储与内存预算

这是当前最关键的实际约束。

假设 `D=1024`。

- `8k` token，`float32`：约 `8,192 * 1,024 * 4 ~= 32 MB / sample`
- `32k` token，`float32`：约 `128 MB / sample`
- `128k` token，`float32`：约 `512 MB / sample`
- `1M` token，`float32`：约 `4.0 GB / sample`

这意味着：

- 按当前默认实现，`1M` hidden-state dump 无法大规模生产
- `1M` 测试建议先从 `50-200` 个样本起步
- 如果后续需要做大规模 `1M` 测试，应该优先把 hidden state 存成 `bf16/fp16`

另外还需要注意：

- trainer 在训练时会构造一个 corrupted student view，因此输入 hidden state 在显存里实际上会接近翻倍
- 对于 `1M`，应默认使用 `batch size = 1`


## 7. 测试阶段划分

### Phase 0：冒烟测试

目标：

- 在正式占用 A800 资源前，先验证端到端链路是通的

动作：

- 使用 `model/Genos_m.tiny_test`
- 在很短序列上跑 extraction
- 跑 1 个 epoch 的 JEPA 训练
- 导出 embedding
- 运行 linear probe

通过标准：

- 整条 pipeline 不需要改代码即可跑通

### Phase 1：短长度 baseline 检查

目标：

- 先建立一个最基础的判断：readout head 是否比简单 pooling 有价值

长度：

- `8k`、`32k`

模型：

- `mean`
- `attention`
- `qformer`

指标：

- JEPA loss 曲线
- linear probe accuracy / MSE
- embedding 方差与塌缩迹象

通过标准：

- `qformer` 在多数任务上优于 `mean`

### Phase 2：扩展到 128k 的长上下文测试

目标：

- 验证在当前仓库能力范围内，这个方法在长长度下依然成立

长度：

- `128k`

消融项：

- chunk size：`64` vs `128`
- queries：`16` vs `32` vs `64`
- span length 范围
- corruption ratio：`0.15` vs `0.30`

通过标准：

- 训练过程稳定
- 性能仍高于简单 baseline
- 没有因为过强 pooling 导致的明显信息塌缩

### Phase 3：1M pilot

目标：

- 验证所宣称的 `1M` context 在这个 readout 方案下是否真的可用

注意：

- 这一阶段是 pilot，不是完整 benchmark
- 一开始只用很小的数据集
- 不要从大规模 `1M` hidden-state dump 多 epoch 训练起步

推荐顺序：

1. 先提取或准备一个小规模 `1M` hidden-state 集合
2. 先只做 readout forward 和 embedding export
3. 在冻结 embedding 上做 linear probe
4. 如果都稳定，再做短程 JEPA 训练，`batch size = 1`

通过标准：

- `1M` export 成功
- embedding 没有塌缩
- probe 明显优于简单 baseline

### Phase 4：鲁棒性与不变性验证

目标：

- 检验 embedding 在部分扰动和裁剪下是否仍然可用

检查项：

- 完整序列与随机 crop 后 embedding 的 cosine similarity
- 不同 `mask_ratio` 下的 corruption robustness
- 同一个长样本相邻窗口之间的一致性

通过标准：

- 表征对于下游使用足够稳定，不会对小幅 view 变化过于敏感


## 8. A800 80G x 8 的推荐用法

由于仓库本身不是分布式训练实现，建议把 8 张卡用于并行实验。

推荐分配：

- GPU0-GPU2：并行跑 Phase 1 baseline
- GPU3-GPU4：跑 Phase 2 的 `128k` 消融
- GPU5-GPU6：跑 hidden-state extraction 分片
- GPU7：跑 `1M` pilot、export 和 probe

如果 hidden states 已经提前准备好：

- 那么 8 张卡都优先用于并行 sweep readout 配置


## 9. 推荐配置覆盖方式

### Phase 1

```bash
python train.py --config configs/v1.yaml --override \
  data.data_root=/path/to/hs_8k \
  data.max_length=8192 \
  training.batch_size=8 \
  training.epochs=10 \
  model.type=qformer \
  model.chunk_pooling.chunk_size=64 \
  model.chunk_pooling.stride=64
```

### Phase 2

```bash
python train.py --config configs/v1.yaml --override \
  data.data_root=/path/to/hs_128k \
  data.max_length=131072 \
  training.batch_size=2 \
  training.epochs=10 \
  model.type=qformer \
  model.num_queries=32 \
  model.chunk_pooling.chunk_size=64 \
  model.chunk_pooling.stride=64
```

### Phase 3

```bash
python train.py --config configs/v1.yaml --override \
  data.data_root=/path/to/hs_1m \
  data.max_length=1048576 \
  data.random_crop=false \
  training.batch_size=1 \
  training.epochs=3 \
  model.type=qformer \
  model.num_queries=32 \
  model.chunk_pooling.chunk_size=64 \
  model.chunk_pooling.stride=64
```

embedding 导出：

```bash
python eval/export_embeddings.py \
  --config configs/v1.yaml \
  --override data.data_root=/path/to/hs_eval data.max_length=131072 \
  --checkpoint /path/to/checkpoint_epoch_10.pt \
  --output /path/to/embeddings.pt \
  --batch-size 4
```

linear probe：

```bash
python eval/linear_probe.py \
  --embeddings /path/to/embeddings.pt \
  --epochs 50 \
  --lr 1e-3 \
  --task auto
```


## 10. 建议记录的指标

每个实验至少记录以下信息：

- train JEPA loss
- variance penalty 和 covariance penalty
- 验证集上的 probe 指标
- 每个 epoch 的运行时间
- 峰值 GPU 显存
- hidden-state 的磁盘占用
- 每个样本的 export 延迟

做长度扩展对比时，必须保证：

- 任务相同
- train/val 切分逻辑相同
- readout family 相同
- 只改变长度


## 11. 最终 Go / No-Go 判定规则

只有满足下面条件，才建议给出 **Go**：

- `qformer` 持续优于简单 pooling
- `128k` 结果稳定且有效
- `1M` pilot 在工程上跑通
- hidden-state 提取的磁盘和时间成本对于目标数据规模是可接受的

如果出现下面任一情况，则建议 **No-Go 或需要改架构**：

- `qformer` 并没有优于 `mean`
- JEPA 训练中 embedding 塌缩
- 从 `32k` 到 `128k` 质量急剧下降
- `1M` extraction 成为整个项目的主导成本


## 12. 最可能的瓶颈

基于当前代码，风险最高的点大概率是：

1. `1M` hidden-state extraction，而不是 readout training
2. `float32` hidden-state 存储开销过大
3. 缺少分布式 extraction / training 支持
4. 除了 linear probe 外，没有 retrieval 或 ranking 类评测

如果前两项成为阻塞点，优先的工程动作应该是：

- 把 hidden state 存成 `bf16/fp16`
- 只保存必要层
- 增加 dataset sharding 和多进程 extraction
- 在 linear probe 之外补一个 retrieval benchmark
