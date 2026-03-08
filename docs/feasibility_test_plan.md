# ContextAgg Feasibility Test Plan

## 1. Scope

This repository does **not** validate full-model language modeling quality by itself. It validates a narrower question:

> Can frozen hidden states from the Genos-m backbone be compressed by a readout JEPA head into a useful sequence-level embedding?

Concretely, the current codebase supports:

- hidden-state extraction from a local HF checkpoint: `data/extract_hidden_states.py`
- JEPA readout training on frozen hidden states: `train.py`
- embedding export: `eval/export_embeddings.py`
- linear probe evaluation: `eval/linear_probe.py`

So the right feasibility target is:

1. the backbone hidden states already contain usable long-context information
2. the readout head can recover that information into `z_seq`
3. the approach remains stable as sequence length grows from `8k -> 32k -> 128k -> 1M`


## 2. Current Repository Boundaries

Before testing, several constraints need to be explicit:

- `train.py` is single-process, single-GPU. There is no DDP/FSDP.
- `data/extract_hidden_states.py` is also single-process, single-GPU.
- `configs/v1.yaml` defaults to `data.max_length=131072`, not `1M`.
- extraction currently saves `last_hidden.float().cpu()`, so hidden states are stored in `float32`
- `HiddenStateDataset` converts loaded states to `float32`
- the repo has a linear-probe evaluator, but no retrieval benchmark script and no long-context serving benchmark

Implication:

- With `A800 80G x 8`, the best immediate use of the 8 GPUs is **parallel experiment execution**, not one distributed training job.
- Full `1M` end-to-end validation is realistic only as a **pilot** unless hidden-state storage and extraction are optimized.


## 3. Why This Project Is Still a Good Feasibility Test

The model checkpoint in this repo declares:

- `hidden_size=1024`
- `num_local_experts=32`
- `num_experts_per_tok=2`
- `max_position_embeddings=1048576`

That matches the background assumption:

- 4.7B total MoE
- 32 experts, 2 active
- active params about 330M
- GQA
- max context 1M

The readout path is computationally plausible for long context because:

- chunk pooling is already implemented in `models/chunk_pooling.py`
- default chunk pooling `64/64` reduces `1,000,000` tokens to about `15,625` pooled tokens
- Q-Former cross-attention complexity is then on `num_queries x pooled_length`, not `raw_length x raw_length`

So the likely bottleneck is **hidden-state extraction and storage**, not the JEPA readout itself.


## 4. Success Criteria

Treat the model as feasible only if all four gates pass.

### Gate A: Engineering feasibility

- extraction, training, export, and probe all run without NaN / inf / shape failures
- checkpoints are produced normally
- embedding export works on validation data

### Gate B: Representation usefulness

- `qformer` is better than a trivial readout baseline (`mean` or `last`) on at least 2 downstream tasks
- linear probe performance is materially above majority-class or random baseline
- JEPA training loss trends down and embeddings do not collapse

### Gate C: Length scaling

- quality degradation from `8k -> 32k -> 128k` is controlled
- the same head family still works at `128k`
- a `1M` pilot can complete export and downstream evaluation on a small held-out set

### Gate D: Resource feasibility

- `128k` training is stable on one `A800 80G`
- `1M` inference / export fits in memory for at least batch size `1`
- disk and runtime are acceptable for a pilot


## 5. Data Design

Because the repo has no bundled dataset, build evaluation data in two layers.

### Layer 1: Synthetic control tasks

These are mandatory because they isolate whether the embedding actually preserves long-range information.

- motif presence classification
- motif count regression
- motif order classification: pattern A before B vs B before A
- long-range interaction classification: two motifs separated by very long distance
- distractor robustness: same signal, increasing irrelevant background

Recommendation:

- each task should have length buckets: `8k`, `32k`, `128k`
- for `1M`, keep only a pilot subset at first
- keep label generation deterministic so failures are easy to interpret

### Layer 2: Real downstream tasks

Choose at least 3 real tasks:

- 2 classification tasks
- 1 regression task

If this project is for biological sequences, the most natural choices are:

- taxonomy or family classification
- phenotype / host / attribute classification
- continuous score regression

Each sample should be saved in the repo-supported format:

```python
{
  "hidden_states": Tensor[L, D],
  "attention_mask": Tensor[L],
  "label": Tensor[...] or scalar
}
```


## 6. Storage and Memory Budget

This is the main practical constraint.

Assume `D=1024`.

- `8k` tokens in `float32`: about `8,192 * 1,024 * 4 ~= 32 MB / sample`
- `32k` tokens in `float32`: about `128 MB / sample`
- `128k` tokens in `float32`: about `512 MB / sample`
- `1M` tokens in `float32`: about `4.0 GB / sample`

That means:

- `1M` hidden-state dumps cannot be produced at scale with the current default path
- a `1M` test should start from `50-200` samples max
- if large-scale `1M` testing is needed, extraction should be changed to `bf16/fp16` storage first

Also note:

- trainer creates a corrupted student view, so raw input states are effectively duplicated in memory during training
- for `1M`, batch size should be assumed to be `1`


## 7. Test Phases

### Phase 0: Smoke test

Goal:

- validate the pipeline end to end on tiny data before spending A800 time

Actions:

- use `model/Genos_m.tiny_test`
- run extraction on very short sequences
- run 1 epoch JEPA training
- export embeddings
- run linear probe

Pass condition:

- the full pipeline finishes without code changes

### Phase 1: Short-length baseline check

Goal:

- establish whether the readout head adds value over trivial pooling

Lengths:

- `8k`, `32k`

Models:

- `mean`
- `attention`
- `qformer`

Metrics:

- JEPA loss curve
- linear probe accuracy / MSE
- embedding variance and collapse symptoms

Pass condition:

- `qformer` beats `mean` on most tasks

### Phase 2: Long-context scaling to 128k

Goal:

- verify the method remains useful at long length with current repo support

Lengths:

- `128k`

Ablations:

- chunk size: `64` vs `128`
- queries: `16` vs `32` vs `64`
- span length range
- corruption ratio: `0.15` vs `0.30`

Pass condition:

- training remains stable
- performance is still above trivial baselines
- no obvious collapse from aggressive pooling

### Phase 3: 1M pilot

Goal:

- validate that the claimed `1M` context is usable for this readout setup

Important:

- this phase is a pilot, not a full benchmark
- use a very small dataset first
- do not start with multi-epoch training on large `1M` dumps

Recommended order:

1. extract or obtain a small `1M` hidden-state set
2. run readout forward and embedding export only
3. run linear probe on frozen embeddings
4. if stable, run short JEPA training with batch size `1`

Pass condition:

- `1M` export succeeds
- embeddings are non-collapsed
- probe beats trivial baseline

### Phase 4: Robustness and invariance

Goal:

- test whether embeddings remain useful under partial corruption and cropping

Checks:

- full sequence vs random crop embedding cosine similarity
- corruption robustness under different `mask_ratio`
- consistency across nearby windows from the same long sample

Pass condition:

- representation is stable enough for downstream use and not hypersensitive to small view changes


## 8. Recommended GPU Allocation on A800 80G x 8

Because the repo is not distributed, use the 8 GPUs for parallel experiments.

Suggested allocation:

- GPU0-GPU2: Phase 1 baselines in parallel
- GPU3-GPU4: Phase 2 `128k` ablations
- GPU5-GPU6: hidden-state extraction shards
- GPU7: `1M` pilot and export / probe

If hidden states are already available:

- use all 8 GPUs for parallel readout sweeps instead


## 9. Recommended Config Overrides

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

Embedding export:

```bash
python eval/export_embeddings.py \
  --config configs/v1.yaml \
  --override data.data_root=/path/to/hs_eval data.max_length=131072 \
  --checkpoint /path/to/checkpoint_epoch_10.pt \
  --output /path/to/embeddings.pt \
  --batch-size 4
```

Linear probe:

```bash
python eval/linear_probe.py \
  --embeddings /path/to/embeddings.pt \
  --epochs 50 \
  --lr 1e-3 \
  --task auto
```


## 10. Metrics to Record

For every run, record at least:

- train JEPA loss
- variance penalty term and covariance penalty term
- probe metric on validation split
- runtime per epoch
- peak GPU memory
- hidden-state disk footprint
- export latency per sample

For length-scaling comparison, always compare:

- same task
- same train/val split logic
- same readout family
- different lengths only


## 11. Final Go / No-Go Rule

Recommend a **Go** decision only if:

- the `qformer` head consistently outperforms trivial pooling
- `128k` is stable and useful
- `1M` pilot succeeds technically
- the disk and runtime cost of hidden-state extraction is acceptable for the intended dataset size

Recommend **No-Go or architecture revision** if any of the following happens:

- `qformer` does not beat `mean`
- embeddings collapse during JEPA training
- quality drops sharply from `32k` to `128k`
- `1M` extraction cost dominates the whole project


## 12. Most Likely Bottlenecks

Based on the current codebase, the highest-risk items are:

1. `1M` hidden-state extraction, not readout training
2. `float32` hidden-state storage size
3. lack of distributed extraction / training support
4. absence of retrieval or ranking evaluation beyond linear probe

If the first two become blockers, the next engineering step should be:

- save hidden states in `bf16/fp16`
- optionally save selected layers only
- add dataset sharding and multi-process extraction
- add a retrieval benchmark beside the linear probe
