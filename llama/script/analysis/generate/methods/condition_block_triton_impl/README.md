# Condition-block Triton implementation

模块划分：

- `core.py`：generation、prompt block summary、condition selection、runner 与 HF attention 接入。
- `page_attention.py`：当前 fast path；融合 representative、selected page、suffix 和 online softmax。
- `legacy.py`：dense、compact SDPA 及旧 Triton 实现，仅用于回归对齐和实验开关。

Fast path 支持 `block_size=16/32`。底层 Tensor Core tile 固定为 16 tokens：

- 16-token selected page 使用一次 MMA tile。
- 32-token selected page 使用两次 MMA tile，并在 kernel 内做 online-softmax 合并。

顶层 `condition_block_triton.py` 只保留兼容导出，现有 CLI/import 不需要修改。

Stats 默认开启，用于输出等效 budget。实现中 stats 计数延迟到样本结束后
materialize，避免 decode 每层每步 `.item()` 同步；纯测速可设置
`CONDITION_BLOCK_SKIP_STATS=1` 关闭。

## Post-prefill StaticCache

长生成下，Triton sparse attention 本身已经比 dense/full attention 小很多，但
HF `DynamicCache` 在每个 decode step 会用 `torch.cat` 追加每层 K/V，profile 里
表现为大量 `aten::cat` / `CatArrayBatchedCopy` / `copy_`，会吃掉 sparse attention
省下来的时间。

当前实现提供一个 opt-in 开关：

```bash
CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1
```

语义：

1. prefill 仍使用默认 DynamicCache + SDPA，避免从一开始使用 StaticCache 导致
   prefill 变慢；
2. prefill 结束后，将 prompt KV 一次性复制进 `StaticCache`；
3. decode 阶段用 `cache_position` 原地写入新 token KV，避免每步沿 sequence 维
   拼接整段 KV。

这个开关会增加一次 prefill 后的 KV 拷贝，并提高显存峰值；因此没有设为默认。
它更适合 64K/128K 这种长上下文、长生成测速。

Sanity check：

```text
LongBench-v2 32K prompt
max_new_tokens = 8
DynamicCache vs post-prefill StaticCache: generated tokens exactly match
```

Latency probe：

```text
model: Llama-3.1-8B-Instruct
dataset: LongBench-v2
max_new_tokens: 128 fixed decode
block_size: 32
eps: 0.1
stats: disabled
selection: compiled

context   old Triton   post-static Triton   full attention
32K       6.05s        5.41s                5.23s
64K       11.67s       10.30s               10.36s
128K      27.61s       24.92s               25.35s
```

KV/cache profiler category:

```text
32K:   3.31s  -> 1.97s
64K:   6.61s  -> 3.78s
128K: 13.17s -> 7.35s
```

推荐长上下文 latency 配置：

```bash
CONDITION_BLOCK_SKIP_STATS=1 \
CONDITION_BLOCK_COMPILE_SELECTION=1 \
CONDITION_BLOCK_TRITON_CHUNK_BLOCKS=64 \
CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1
```

## Decode-only and dummy-selection profiling

为了避免 prefill 掩盖 decode 阶段差异，`longbench_v2_latency.py` 增加了：

```bash
--decode-only-timing
--profile-decode-only
```

实现方式是不改变原 generate 路径，只在第一次 `model.forward` 返回后开始计时
/ profiler。第一次 forward 对应 long-context prefill，因此输出里的
`decode_only_seconds` 可以近似表示只生成后续 token 的耗时。

示例：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/scratch1/liankewei/interattn \
CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1 \
python -m llama.script.analysis.generate.longbench_v2_latency \
  --device cuda:0 \
  --methods full condition_block_triton \
  --contexts 32768 65536 131072 \
  --max-new-tokens 128 \
  --samples 1 --warmup 1 \
  --condition-block-size 32 \
  --condition-eps 0.1 \
  --skip-stats \
  --compile-selection \
  --fixed-decode \
  --decode-only-timing \
  --output /tmp/lb2_decode_only_32k_128k.jsonl
```

Decode-only latency, `max_new_tokens=128`, `block_size=32`, `eps=0.1`：

```text
context   full decode   Triton decode   Triton vs full
32K       2.828s        2.979s          0.95x
64K       4.020s        3.956s          1.02x
128K      6.301s        5.704s          1.10x
```

32K decode-only profiler breakdown shows why 32K is not yet favorable:

```text
full:
  model_gemm_gemv        55.4%
  kv_cache_copy_index    26.0%
  flash_sdpa_attention   13.0%

Triton:
  model_gemm_gemv        53.9%
  kv_cache_copy_index    34.6%
  condition_selection     2.1%
  sparse_finalize_attn    2.7%
  sparse_reduce           0.2%
```

At 32K, full SDPA decode attention is already small enough that the extra
condition-selection and custom sparse-path overhead can erase the saved
attention work.

Two synthetic/diagnostic scripts were added:

- `condition_block_stage_latency.py`: synthetic microbench for selection,
  dummy selected sparse attention, and full SDPA decode attention.
- `condition_block_dummy_e2e_latency.py`: end-to-end generate with production
  selection replaced by a fixed dummy selected-block ratio.

Dummy-selection 32K decode-only result, `max_new_tokens=128`：

```text
config              decode-only   vs production
production          2.973s        1.00x
dummy 0% selected   2.800s        1.06x
dummy 10% selected  3.128s        0.95x
dummy 25% selected  3.163s        0.94x
```

Interpretation:

- dummy selection skips query-dependent condition work:
  `q @ k_bar`, `q @ k_max/k_min`, delta, condition score, selected-mask
  decision;
- it still computes attention over unselected representatives and selected
  block tokens;
- at 32K, even making selection free only improves decode by about 7% in the
  best 0%-selected case;
- with 10%/25% selected blocks, selected-token attention cost outweighs the
  removed selection cost.

Conclusion: for 32K single-batch decode, selection is not the dominant bottleneck.
The largest profiler buckets are model GEMM/GEMV and KV/cache copy/index. Sparse
attention becomes useful at longer contexts, where full decode attention grows
linearly while representative attention grows much more slowly.

## IO-bound Optimization Analysis Plan

Decode attention is largely IO-bound: full attention repeatedly reads all
historical KV from HBM, while condition-block should read representatives for
unselected blocks and only read token KV for selected blocks. The optimization
question is therefore not only whether sparse attention computes fewer scores,
but whether the whole decode path actually reduces HBM traffic enough to beat
Flash/SDPA plus model-side overhead.

Use the following three-step analysis before choosing the next kernel change.
All latency comparisons should use `decode_only_seconds`, not total
`generation_seconds`, unless explicitly studying prefill or end-to-end serving
latency.

### Step 1: Condition selection efficiency

Measure the query-dependent cluster selection path with
`condition_block_stage_latency.py`.

Track:

- `selection_stats_reduce`: computes the stats needed by the condition formula.
- `selection_materialize`: produces the selected-block mask.
- Full SDPA decode attention as a reference point.

The selection path includes:

```text
q @ k_bar
q @ k_max / k_min
delta
condition score
selected mask decision
```

Representative/cluster centers themselves (`k_bar`, `v_bar`, `k_min`, `k_max`)
are prompt-side summaries and are cached once per layer. The repeated per-token
cost is deciding which cached clusters the current query should expand.

Success criteria:

- Determine whether selection grows with context length.
- Compare selection latency against full SDPA decode attention at 32K/64K/128K.
- If selection is close to, or larger than, full SDPA attention at a context
  length, sparse attention will need selection/attention fusion to win there.

### Step 2: Dummy-selection sparse attention upper bound

Use dummy selected-block ratios to isolate the attention kernel after cluster
selection is already known.

Recommended ratios:

```text
0%, 5%, 10%, 25%
```

Use:

- `condition_block_stage_latency.py` for synthetic sparse-attention upper bound.
- `condition_block_dummy_e2e_latency.py` for generate-path behavior with
  production selection replaced by fixed dummy masks.

Dummy selection skips:

```text
q @ k_bar for condition selection
q @ k_max / k_min
delta
condition score
selected mask decision
```

Dummy selection still computes:

```text
representative attention for unselected blocks: q @ k_bar
token attention for selected blocks: q @ k_token
exact generated-suffix attention: q @ k_suffix
model forward, KV cache update, hooks, logits
```

Success criteria:

- If dummy `0%` selected is still not clearly faster, the bottleneck is not
  condition selection alone.
- If dummy `10%` or `25%` selected becomes slower, selected-token KV IO already
  consumes the benefit of skipping selection.
- If dummy sparse attention is fast in synthetic microbench but weak in
  generate, the remaining bottleneck is outside the sparse-attention math.

### Step 3: End-to-end decode-only attribution

Run production decode-only profiling on the original generate path:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/scratch1/liankewei/interattn \
CONDITION_BLOCK_SKIP_STATS=1 \
CONDITION_BLOCK_COMPILE_SELECTION=1 \
CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1 \
python -m llama.script.analysis.generate.longbench_v2_latency \
  --device cuda:0 \
  --methods full condition_block_triton \
  --contexts 32768 65536 131072 \
  --max-new-tokens 128 \
  --samples 1 --warmup 1 \
  --condition-block-size 32 \
  --condition-eps 0.1 \
  --skip-stats \
  --compile-selection \
  --fixed-decode \
  --decode-only-timing \
  --profile \
  --profile-decode-only \
  --output /tmp/lb2_decode_profile.jsonl
```

Also run dummy-selection decode-only profiling:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.generate.condition_block_dummy_e2e_latency \
  --device cuda:0 \
  --contexts 32768 \
  --max-new-tokens 128 \
  --samples 1 --warmup 1 \
  --condition-block-size 32 \
  --selected-ratios 0 0.1 0.25 \
  --include-production \
  --fixed-decode \
  --decode-only-timing \
  --profile \
  --profile-decode-only \
  --output /tmp/dummy_decode_profile_32k.jsonl
```

Fixed setting:

```text
max_new_tokens = 128
block_size = 32
eps = 0.1
stats disabled
compiled selection enabled
post-prefill StaticCache enabled
```

Important profiler buckets:

- `flash_sdpa_attention`: full-attention decode kernel cost.
- `condition_selection`: query-dependent cluster selection.
- `sparse_finalize_attention`: representative + selected-token sparse attention.
- `sparse_reduce`: partial softmax reduce across block chunks.
- `kv_cache_copy_index`: KV cache update/read/slice/copy/index traffic.
- `model_gemm_gemv`: model linear layers, MLP, projections, lm head.

Success criteria:

- Compare `full` vs `condition_block_triton` using `decode_only_seconds`.
- Compare production vs dummy `0%/10%/25%` selected using the same metric.
- Use profiler percentages only for attribution; profiler wall time is slower
  than normal execution and should not replace clean latency runs.

### Decision criteria / next optimization target

- If `condition_selection` dominates: optimize selection, reduce metadata reads,
  or fuse selection more deeply with sparse attention.
- If `sparse_finalize_attention` dominates: optimize selected page loading,
  selected-token IO, and paged-attention-style kernels.
- If `sparse_reduce` dominates: reduce chunk count or fuse/rework partial
  softmax reduction.
- If `kv_cache_copy_index` dominates: optimize cache layout, generated-suffix
  slicing, StaticCache use, and avoid extra contiguous/copy/index operations.
- If `model_gemm_gemv` dominates: single-batch attention optimization has limited
  remaining headroom; consider larger batch, CUDA graph / lower eager dispatch,
  or accept that this context length is close to the decode-path ceiling.

Current data suggests:

- At 32K, full SDPA decode attention is already small; dummy `0%` selected only
  improves production decode by about 6%.
- At 64K, Triton decode is roughly break-even.
- At 128K, Triton decode shows a clearer benefit because full attention KV IO
  grows linearly with context while representative attention grows much more
  slowly.

### Latest IO-bound analysis results

Fresh outputs:

```text
/tmp/cb_io_stage_32k_128k.jsonl
/tmp/cb_io_decode_32k_128k.jsonl
/tmp/cb_io_dummy_32k_128k.jsonl
/tmp/cb_io_decode_profile_32k.jsonl
```

Stage microbench:

| context | selection materialize | full SDPA attn | dummy sparse 0% | dummy sparse 10% | dummy sparse 25% |
|---:|---:|---:|---:|---:|---:|
| 32K | 0.069 ms | 0.080 ms | 0.031 ms | 0.031 ms | 0.031 ms |
| 64K | 0.068 ms | 0.218 ms | 0.031 ms | 0.031 ms | 0.040 ms |
| 128K | 0.073 ms | 0.403 ms | 0.031 ms | 0.043 ms | 0.111 ms |

Decode-only full vs Triton:

| context | full decode | Triton decode | Triton vs full |
|---:|---:|---:|---:|
| 32K | 2.828s | 2.979s | 0.95x |
| 64K | 4.020s | 3.956s | 1.02x |
| 128K | 6.301s | 5.704s | 1.10x |

Dummy-selection decode-only:

| context | production | dummy 0% | dummy 10% | dummy 25% |
|---:|---:|---:|---:|---:|
| 32K | 2.973s | 2.800s (1.06x) | 3.128s (0.95x) | 3.163s (0.94x) |
| 64K | 3.957s | 3.676s (1.08x) | 4.040s (0.98x) | 4.126s (0.96x) |
| 128K | 5.745s | 5.397s (1.06x) | 5.836s (0.98x) | 6.016s (0.95x) |

Takeaways:

- The synthetic sparse attention upper bound is strong: dummy sparse attention
  is much cheaper than full SDPA at 64K/128K.
- In production decode, that win is partly offset by KV/cache traffic and model
  GEMM/GEMV, which dominate the 32K profile.
- Dummy `0%` confirms that making selection free only gives a small decode gain;
  dummy `10%/25%` confirms selected-token IO can quickly eat the selection
  savings.

## Kernel 级 decode 归因与逐步 clone 修复

`condition_block_decode_attribution.py` 在 bucket profile 之上补三个 bucket
回答不了的数字：每个 decode step 的 GPU busy vs wall、每步 kernel launch 数、
以及单 kernel 耗时（只统计 device 侧 event，避免 CPU op 与 kernel 重复计数）。
wall 时间取自不开 profiler 的干净 run，kernel 时间取自单独的 profiled run。

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/scratch1/liankewei/interattn \
CONDITION_BLOCK_SKIP_STATS=1 CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1 \
python -m llama.script.analysis.generate.condition_block_decode_attribution \
  --device cuda:0 --methods full condition_block_triton \
  --contexts 32768 65536 131072 --max-new-tokens 128
```

输出：`llama/result/generate/cb_attr_{full,triton,triton_fixed}.jsonl`。

### 发现：fused 路径每个 decode step 都 clone 整个 blocked prompt KV

用 `record_shapes` + stack profile 定位到 Triton 路径最大的 copy 是对
`[8, n_blocks, 32, 128]` 的 `aten::contiguous`，每层每步各两次：
`k_block_attn` / `v_block_attn` 每层只构建一次并缓存在 `prompt_prefix`，
但它们是 KV cache buffer 的非连续 view；fused kernel 启动时每步都对它们调用
`.contiguous()`，等于每步把整个 blocked prompt K/V 各 clone 一遍
（32K 时每层 2 x 67MB）：32K 单次 98.5us（占 GPU busy 30.7%），64K 195us
（44.9%）—— 随 context 线性增长，超过了 sparse attention 省下的全部时间。

修复：finalize kernel 增加 `k_block_head_stride` / `v_block_head_stride`，
直接按 stride 读 cache view 里的 token page（与 suffix 已有的做法一致）；
`_ensure_paged_layout` 对意外 layout 保留一次性 contiguous 兜底。
每步零拷贝、零额外显存。

Sanity check：

- kernel 单元检查：非连续 view 与 contiguous copy 两种输入，输出和 selected
  mask 逐位一致（prompt 长度整除与带 padding 两种情况）；
- e2e：修复前后生成 token 完全一致（32K/64K post-prefill StaticCache、
  32K DynamicCache；LongBench-v2，32 tokens）。

### Decode-only latency 修复前后（128 fixed tokens, block 32, eps 0.1）

| context | full (HF DynamicCache) | Triton 修复前 | Triton 修复后 | 修复后 vs full |
|---:|---:|---:|---:|---:|
| 32K  | 22.26 ms/step | 23.57 ms/step (0.94x) | 17.37 ms/step | **1.28x** |
| 64K  | 31.73 ms/step | 30.64 ms/step (1.04x) | 18.04 ms/step | **1.76x** |
| 128K | 49.57 ms/step | 44.9 ms/step (1.10x)  | 19.70 ms/step | **2.52x** |

修复后 decode step 随 context 基本持平（32K→128K 只从 17.4 涨到 19.7 ms/step），
这正是该方法应有的 scaling。

Baseline 说明：HF `full` baseline 自身在 DynamicCache `torch.cat` 上消耗越来越
大（32K/64K/128K 分别占其 GPU busy 的 24%/34%/46%）。若与去掉 cat 的理想 full
baseline（17.4 / 21.5 / 27.6 ms/step）对比，修复后为 1.00x / 1.19x / 1.40x。
HF 自带的 `cache_implementation="static"` generate 实测反而慢 3.6 倍，
不能当作更强的 baseline。

### 修复后归因（Triton fixed）

| per step | 32K | 64K | 128K |
|---|---:|---:|---:|
| decode wall | 17.37 ms | 18.04 ms | 19.70 ms |
| GPU busy（占比） | 14.74 ms (85%) | 15.60 ms (86%) | 17.30 ms (88%) |
| model_gemm_gemv | 73.4% | 69.2% | 61.9% |
| condition_selection | 5.7% | 8.3% | 12.9% |
| sparse_finalize_attention | 7.3% | 7.8% | 8.2% |
| kv_cache_copy_index | 4.2% | 4.5% | 5.1% |
| kernel launches | 1456 | 1456 | 1456 |

128K 时每层每步 sparse 路径约 117us（selection stats 68us + finalize 44us +
reduce 约 4us），对比 full attention 的 flash kernel 399us；sparse 各 kernel
中只有 selection-stats 仍随 context 明显增长。

生产 selected ratio（stats 开，eps 0.1）：32K/64K/128K 分别只选中
1.4% / 0.7% / 0.4% 的 block（等效 budget 7.5% / 6.8% / 6.6%），selected page
的 IO 可以忽略；sparse 路径主要开销在 representative 与 summary 的读取。

### 已排除：BF16 block summaries

`condition_block_stage_latency.py --summary-dtype bfloat16` 把
`k_bar/k_max/k_min/v_bar` 转成 BF16（kernel 加载后在寄存器转 FP32，无需改
kernel）。所有 context 下均为 0.95-1.03x —— selection kernel 在这些尺寸下
不是带宽瓶颈，砍半 summary IO 没有收益。不采用。

### 剩余 decode 优化空间（按优先级）

1. Model GEMV/GEMM 是天花板：约 10.8 ms/step，与 context 无关，有效带宽约
   1.5 TB/s（约峰值 83%）。再往下要靠权重量化、更大 batch 或 speculative
   decoding，而不是 attention 侧。
2. Host/launch 开销：每步 idle 2.4-2.6 ms（12-15%），约 1456 次 kernel
   launch。下一步最自然的优化是把整个 decode step 做成 CUDA graph。
   （已实现，见下节。）
3. Sparse 路径融合：把 selection-stats 与 finalize-attention 合并
   （finalize 本来就从 cache 重读 s/delta），每层可省一次随 context 增长的
   pass 和一次 launch；收益上限是 condition_selection 的 5.7-12.9%。
4. 残留：每步一次、随 context 成比例的 copy（32K 102us → 128K 389us，
   约占 busy 1-2%），疑似 HF 侧 mask/cache 处理；优先级低。

## CUDA graph decode（消除 host/launch 开销）

```bash
CONDITION_BLOCK_CUDA_GRAPH=1   # 需配合 CONDITION_BLOCK_SKIP_STATS=1
```

把整个 decode step（model forward + argmax + 状态推进）capture 成一张 CUDA
graph，之后每生成一个 token 只做一次 `graph.replay()`，replay 阶段完全不执行
Python，host/launch 开销压缩为单次 graph 启动。

实现要点：

- fused kernel 的 suffix 长度从标量参数改为 `suffix_len_ptr`（device 上的
  int32），graph 内每步 `add_(1)` 推进，kernel launch 无需重录；eager 路径由
  `reset_step` 每步 `fill_` 同一个 scalar，一步一个小 kernel，行为不变。
- runner 增加 `static_suffix` 模式：直接把 StaticCache 的整块 K/V view 交给
  kernel（shape 每步不变，capture 需要），有效 suffix 长度由 `suffix_len_dev`
  控制，不再按 `pos` 做 Python 切片。
- `input_ids` / `cache_position` / 输出 token 槽位都是固定 device buffer，
  argmax 结果 in-graph 写回 `input_ids`，`cache_position`、step index、
  suffix 长度 in-graph 自增——replay 之间 host 侧零工作。
- decode 期间跳过 HF 的 causal mask 构建（`_no_causal_mask_context`）：
  所有层都走 sparse 路径本来就不读 mask，而 HF `eager_mask` 里的
  `torch.tensor(0.0, device=...)` 是 H2D 拷贝，capture 期间非法。
- prefill 不变（SDPA + DynamicCache），随后一次性搬进 StaticCache；前 3 个
  decode step 以 eager 方式跑（构建 prompt summary、预热 Triton JIT、按
  torch.cuda.graphs recipe 在 side stream 上稳定 allocator），之后 capture 一次、
  replay 其余 token。
- 停止条件改为生成后截断：greedy 下与逐 step early-stop 输出完全一致。
- 限制：`CONDITION_BLOCK_SKIP_STATS=1`、`full_attention_layers=0`、fused
  stage2（block 16/32）、batch 1。

Sanity check：

- 重构后 eager 路径与修复版基线 token 完全一致（32K/64K StaticCache、
  32K DynamicCache）；
- CUDA graph 与 eager 生成 token 完全一致（32K/64K，LongBench-v2）。

Decode-only（128 fixed tokens，含一次性 capture 与 KV 搬移开销，
均摊在 127 步里）：

| context | full | Triton eager（修复后） | Triton CUDA graph | graph vs full |
|---:|---:|---:|---:|---:|
| 32K  | 22.26 ms/step | 17.37 ms/step | 15.43 ms/step | **1.44x** |
| 64K  | 31.73 ms/step | 18.04 ms/step | 16.41 ms/step | **1.93x** |
| 128K | 49.57 ms/step | 19.70 ms/step | 18.31 ms/step | **2.71x** |

GPU busy 占比从 eager 的 85-88% 提高到约 94%，且 graph 模式的 busy 本身也
略低于 eager（decode 期间不再构建 causal mask）。剩余 wall-busy 差主要是
均摊的一次性 capture/KV 搬移。到此 decode 的主导项只剩 model GEMV
（~10.8 ms/step，权重带宽约 83% 峰值），attention 侧优化空间基本吃完。
## Kernel 带宽效率优化（4 kernel → 3 + launch 配置调优）

CUDA graph 之后 profiling 显示 attention 侧 kernel 只跑到峰值带宽的
9-17%（full attention 的 flash kernel 是 66-92%）：读的数据太少
（每步 190-580 MB），落入 latency/occupancy-bound 区间，launch 配置
（BLOCK=16、4 warps）也偏保守。按 read-volume 理论上限
1/(s + 2/32) ≈ 13-15x 的 attention 加速，实际只兑现 1.3-3.5x，
差距全在带宽效率上。本节改动：

1. **selection-reduce kernel 折叠进 finalize**：normalizer 的
   per-chunk partial（4 × rows × n_chunks，~KB 级，L2 常驻）由每个
   finalize program 在寄存器里自行归约，省掉每层一次 1.2-1.4us 的
   纯 launch 开销 kernel。归约顺序与原 reduce kernel 逐位一致。
   生产路径 kernel 数 4 → 3（stats → finalize → stage2）。
2. **selected page 单 tile 化**：32-token page 原来按两个 16-token
   MMA tile 串行消费（受 BLOCK_N=16 耦合），现在 page tile 与 chunk
   宽度解耦，一个 page 一次 `tl.dot`；page 循环也从"每个 local block
   一个分支"改成 cumsum 提取，只访问真正选中的 page。
3. **launch 配置可调 + 按硬件 sweep**：`CONDITION_BLOCK_SELECT_CHUNK`
   / `SELECT_WARPS` / `FINALIZE_CHUNK` / `FINALIZE_WARPS`。cold-L2
   microbench（8 组输入轮换，模拟每层 GEMV 冲掉 L2）sweep 结论：
   - stats kernel 最优 warp 数随 context 反转：32K（n_blocks≤1024）
     8 warps 最快，64K/128K 2 warps 快约 1.5x → 默认按 n_blocks 自适应。
   - finalize 最优 BLOCK_N=32、4 warps（n_chunks 减半，stage2 partial
     读写也减半）。
   - 32K 整条路径贴着 ~55us/层 的 latency floor，配置几乎不影响；
     64K/128K cold-L2 全路径分别提速 1.21x / 1.35x。
4. **BF16 block summaries（可选 flag）**：
   `CONDITION_BLOCK_SUMMARY_DTYPE=bfloat16` 把 k_bar/k_max/k_min/v_bar
   存成 BF16，summary 读量减半。之前微基准测不出收益是因为 kernel 还在
   latency-bound 区；配置调优后 128K stats 再 -19%。默认仍为 FP32
   （selection 边界 block 可能翻转，LongBench 质量是在 FP32 下验证的）。

Sanity check（全部通过）：

- 合成数据（异质 block、混合选中率、pad/非 pad、suffix 1/33/128）：
  selection mask 与 eager 参考完全一致；输出 vs legacy stage2 参考
  max diff ~2e-4（BF16 量级）。
- 非连续 cache view vs contiguous copy：输出逐位一致。
- 真实模型 + CUDA graph：新 kernel（含新默认配置）与旧 kernel 生成
  token 完全一致（32K/64K，LongBench-v2）。

Decode-only 结果（128 fixed tokens，CUDA graph，eps=0.1，block 32）：

| context | attention us/step 旧→新 | ms/step 旧→新 | vs full e2e | vs full attention 相 |
|---:|---:|---:|---:|---:|
| 32K  | 1978 → 1691 | 15.43 → 15.12 | 1.44x → **1.47x** | 1.32x → **1.54x** |
| 64K  | 2586 → 2016 | 16.41 → 15.76 | 1.93x → **2.01x** | 2.80x → **3.59x** |
| 128K | 3781 → 2897 | 18.31 → 17.34 | 2.71x → **2.86x** | 3.54x → **4.62x** |

再加 `CONDITION_BLOCK_SUMMARY_DTYPE=bfloat16`：attention 降到
1553/1736/2472 us/step（vs full attention 相 1.68x/4.17x/5.41x），
e2e 14.95/15.51/17.03 ms/step（1.49x/2.05x/2.91x）。

per-kernel（us/step，旧 → 新 FP32）：stats 2175→1492（128K，效率
41%→60% 峰值带宽）、finalize 1069→867（32K）、selection-reduce
39-43→0、stage2 150→85（128K）。32K 的 stats（770us）仍在 latency
floor 附近，是剩余 attention 开销里最大的一块；进一步压缩需要减少
summary 读量本身（如 k_max/k_min 换成标量 radius，把 selection 读量
从 4 vector/block 降到 2，read-bound 上限从 ~14x 提到 ~26x）。
