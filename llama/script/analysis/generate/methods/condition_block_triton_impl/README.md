# Condition-block Triton implementation

模块划分：

- `core.py`：generation、prompt block summary、condition selection、runner 与 HF attention 接入。
- `page_attention.py`：当前 fast path；融合 representative、selected page、suffix 和 online softmax。
- `legacy.py`：dense、compact SDPA 及旧 Triton 实现，仅用于回归对齐和实验开关。

Fast path 支持 `block_size=16/32`。底层 Tensor Core tile 固定为 16 tokens：

- 16-token selected page 使用一次 MMA tile。
- 32-token selected page 使用两次 MMA tile，并在 kernel 内做 online-softmax 合并。

顶层 `condition_block_triton.py` 只保留兼容导出，现有 CLI/import 不需要修改。
