import argparse

import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model path, e.g. meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama-2-7b-hf",
        help="Short model name used in kv file naming",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name used in kv/result path",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger"],
        help="Mask generation strategy",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index used in kv file naming",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model loading dtype",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10000,
        help="Optimization steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="logits_kl",
        choices=["logits_kl", "v_l2", "v_kl"],
        help="Training loss: logits-space KL, V-space L2, or V-space KL",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to process",
    )
    layer_group.add_argument(
        "--all-layers",
        action="store_true",
        help="Process all transformer layers",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.25, 0.5, 1],
        help="Budget list",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run full-model sanity check by reinjecting alpha* and reporting final KL",
    )
    return parser.parse_args()


def str_to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")
