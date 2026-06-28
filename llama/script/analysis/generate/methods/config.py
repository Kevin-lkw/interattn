from dataclasses import dataclass


@dataclass
class GenerationMethod:
    name: str
    kind: str
    budget: float | None = 1.0
    max_new_tokens: int = 32
    full_attention_layers: int = 0
    condition_eps: float = 1.0
    condition_block_size: int | None = None
    condition_delta_mode: str = "range_bound"
    quest_page_size: int | None = None
    kvpress_window_size: int = 64
    kvpress_kernel_size: int = 5
    kvpress_alpha_safeguard: float = 0.20
    kvpress_sink_tokens: int = 4

    @property
    def compression_ratio(self):
        if self.budget is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(self.budget)))


def build_method(args):
    budget = None if args.budget is None else float(args.budget)
    quest_page_size = args.quest_page_size
    if args.method == "quest":
        if budget is not None:
            quest_page_size = max(1, int(round(1.0 / budget)))
        elif quest_page_size is not None:
            budget = 1.0 / int(quest_page_size)
    elif args.method not in {"condition_block", "condition_block_triton"} and budget is None:
        budget = 1.0
    return GenerationMethod(
        name=args.method,
        kind=args.method,
        budget=budget,
        max_new_tokens=int(args.max_new_tokens),
        full_attention_layers=int(args.full_attention_layers),
        condition_eps=float(args.condition_eps),
        condition_block_size=args.condition_block_size,
        condition_delta_mode=args.condition_delta_mode,
        quest_page_size=None if quest_page_size is None else int(quest_page_size),
        kvpress_window_size=int(args.kvpress_window_size),
        kvpress_kernel_size=int(args.kvpress_kernel_size),
        kvpress_alpha_safeguard=float(args.kvpress_alpha_safeguard),
        kvpress_sink_tokens=int(args.kvpress_sink_tokens),
    )


def add_method_args(parser):
    parser.add_argument(
        "--method",
        default="full",
        choices=[
            "full",
            "kvpress_snapkv",
            "kvpress_adakv_snapkv",
            "kvpress_streamllm",
            "attention_topk",
            "h2o",
            "condition_block",
            "condition_block_triton",
            "quest",
        ],
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--full-attention-layers", type=int, default=0)
    parser.add_argument("--condition-eps", type=float, default=1.0)
    parser.add_argument("--condition-block-size", type=int, default=None)
    parser.add_argument(
        "--condition-delta-mode",
        choices=["range_bound"],
        default="range_bound",
    )
    parser.add_argument("--quest-page-size", type=int, default=None)
    parser.add_argument("--kvpress-window-size", type=int, default=64)
    parser.add_argument("--kvpress-kernel-size", type=int, default=5)
    parser.add_argument("--kvpress-alpha-safeguard", type=float, default=0.20)
    parser.add_argument("--kvpress-sink-tokens", type=int, default=4)
    return parser
