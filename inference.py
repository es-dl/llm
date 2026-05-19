from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from tokenizer import BytePairTokenizer
from train import MiniQuadtrix, ROOT, row, rule, blank, header, success, W, ARROW


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a Mini Quadtrix BPE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mini-quadtrix-bpe.pt")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def normalize_config(config):
    required = {
        "block_size": 8192,
        "n_embd": 6144,
        "n_head": 48,
        "n_layer": 48,
        "dropout": 0.0,
    }
    config = dict(config or {})
    for key, value in required.items():
        config.setdefault(key, value)
    return SimpleNamespace(**config)


def load_checkpoint(path: Path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a checkpoint saved by train.py with keys: model, config, vocab_size")
    return checkpoint


def resolve_tokenizer_path(args, checkpoint):
    if args.tokenizer is not None:
        return args.tokenizer

    tokenizer_path = checkpoint.get("tokenizer")
    if tokenizer_path:
        return Path(tokenizer_path)

    return ROOT / "tokenizer" / "tokenizer.json"


def generate_reply(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        encoded_prompt = [0]

    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    new_tokens = output_ids[0][len(encoded_prompt) :].tolist()
    return tokenizer.decode(new_tokens).strip()


def chat(model, tokenizer, args, device):
    header("INFERENCE", "quit / exit / q -> end session")
    blank()

    while True:
        try:
            prompt = input(f"  user  {ARROW} ").strip()
        except (EOFError, KeyboardInterrupt):
            blank()
            success("Session ended.")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            blank()
            success("Session ended.")
            break
        if not prompt:
            continue

        response = generate_reply(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        blank()
        print(f"  Model {ARROW} {response}")
        blank()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = normalize_config(checkpoint.get("config"))
    vocab_size = int(checkpoint.get("vocab_size"))

    tokenizer_path = resolve_tokenizer_path(args, checkpoint)
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    if int(tokenizer.vocab_size) != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} does not match checkpoint vocab size {vocab_size}"
        )

    model = MiniQuadtrix(cfg, vocab_size, device).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"{'mini-quadtrix-bpe':^{W}}")
    blank()
    row("Device", device)
    row("Checkpoint", args.checkpoint)
    row("Tokenizer", tokenizer_path)
    row("Vocab size", vocab_size)
    row("Block size", cfg.block_size)
    row("Layers", cfg.n_layer)
    row("Heads", cfg.n_head)
    row("Embedding dim", cfg.n_embd)
    rule()

    if args.prompt is not None:
        response = generate_reply(
            model,
            tokenizer,
            args.prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        blank()
        print(response)
        return

    chat(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
