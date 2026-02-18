"""
Compute scaling parameter count for a given model shape.
"""

import argparse
import json
from datetime import datetime, timezone

import torch


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_scaling_params(depth, aspect_ratio, head_dim, n_kv_head, max_seq_len, window_pattern):
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = model_dim // head_dim
    if n_kv_head is None:
        n_kv_head = n_head
    if n_kv_head <= 0:
        raise ValueError(f"n_kv_head must be > 0 (got {n_kv_head})")
    if n_kv_head > n_head:
        raise ValueError(f"n_kv_head must be <= n_head (got n_kv_head={n_kv_head}, n_head={n_head})")
    if n_head % n_kv_head != 0:
        raise ValueError(
            f"n_head must be divisible by n_kv_head for GQA (got n_head={n_head}, n_kv_head={n_kv_head})"
        )

    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)

    param_counts = model.num_scaling_params()
    n_scaling_params = int(param_counts["transformer_matrices"] + param_counts["lm_head"])

    return {
        "timestamp_utc": utc_timestamp(),
        "depth": int(depth),
        "aspect_ratio": int(aspect_ratio),
        "head_dim": int(head_dim),
        "n_kv_head": int(n_kv_head),
        "max_seq_len": int(max_seq_len),
        "window_pattern": window_pattern,
        "vocab_size": int(vocab_size),
        "model_dim": int(model_dim),
        "n_head": int(n_head),
        "param_counts": {k: int(v) for k, v in param_counts.items()},
        "n_scaling_params": int(n_scaling_params),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Lane B n_scaling_params for a model shape.")
    parser.add_argument("--depth", type=int, default=12, help="transformer depth")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128, help="attention head dimension")
    parser.add_argument("--n-kv-head", type=int, default=None, help="number of KV heads (default: same as n_head)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
    parser.add_argument("--window-pattern", type=str, default="L", help="attention window pattern")
    parser.add_argument("--output-json", type=str, default="", help="optional output JSON path")
    args = parser.parse_args()

    result = compute_scaling_params(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        n_kv_head=args.n_kv_head,
        max_seq_len=args.max_seq_len,
        window_pattern=args.window_pattern,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")

    print(result["n_scaling_params"])


if __name__ == "__main__":
    main()
