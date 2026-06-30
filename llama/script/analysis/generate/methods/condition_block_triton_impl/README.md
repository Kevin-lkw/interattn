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
