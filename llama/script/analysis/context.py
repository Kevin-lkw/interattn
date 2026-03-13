from dataclasses import dataclass
from typing import Any


@dataclass
class RunContext:
    model: Any
    tokenizer: Any
    rope_qkv: Any
    inputs: Any
    outputs: Any
    attn_output: Any
    layer_input: Any
    gt_label: Any
    model_config: Any
    dtype: Any
    device: str
