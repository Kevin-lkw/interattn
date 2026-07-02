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
32K       2.802s        2.979s          0.94x
64K       4.021s        3.955s          1.02x
128K      6.301s        5.697s          1.11x
```

32K decode-only profiler breakdown shows why 32K is not yet favorable:

```text
full:
  model_gemm_gemv        55.4%
  kv_cache_copy_index    25.9%
  flash_sdpa_attention   13.0%

Triton:
  model_gemm_gemv        61.6%
  kv_cache_copy_index    27.6%
  condition_selection     4.0%
  sparse_finalize_attn    1.8%
  sparse_reduce           0.1%
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
production          3.010s        1.00x
dummy 0% selected   2.804s        1.073x
dummy 10% selected  3.142s        0.958x
dummy 25% selected  3.179s        0.947x
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
