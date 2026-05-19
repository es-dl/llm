from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import BytePairTokenizer


W = 78
DOUBLE = "=" * W
SINGLE = "-" * W
TICK = "best"
ARROW = ">"
ROOT = Path(__file__).resolve().parent


def log(message=""):
    print("" if message == "" else message)


def header(title, subtitle=""):
    log()
    log(DOUBLE)
    log(f"  {title}")
    if subtitle:
        log(f"  {subtitle}")
    log(DOUBLE)


def row(label, value="", unit="", note=""):
    label_col = f"  {label:<28}"
    value_col = f"{str(value):<20}"
    unit_col = f"{unit:<8}"
    note_col = f"  {note}" if note else ""
    log(f"{label_col}{value_col}{unit_col}{note_col}")


def rule():
    log(f"  {SINGLE}")


def blank():
    log()


def success(msg):
    log(f"  ok  {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini Quadtrix with a custom byte-pair tokenizer.")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("data", ROOT / "data" / "input.txt")))
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "tokenizer.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mini-quadtrix-bpe.pt")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-train-chars", type=int, default=5_000_000)
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=6144)
    parser.add_argument("--n-head", type=int, default=48)
    parser.add_argument("--n-layer", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--generate-tokens", type=int, default=200)
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.vocab_size = min(args.vocab_size, 512)
        args.tokenizer_train_chars = min(args.tokenizer_train_chars, 100_000)
        args.batch_size = 2
        args.block_size = 128
        args.max_iters = 2
        args.eval_interval = 1
        args.eval_iters = 1
        args.n_embd = 128
        args.n_head = 4
        args.n_layer = 2

    return args


def load_or_train_tokenizer(args, text):
    if args.tokenizer.exists() and not args.retrain_tokenizer:
        tokenizer = BytePairTokenizer.load(args.tokenizer)
        return tokenizer, "loaded"

    train_text = text[: args.tokenizer_train_chars]
    tokenizer = BytePairTokenizer.train(train_text, vocab_size=args.vocab_size)
    tokenizer.save(args.tokenizer)
    return tokenizer, "trained"


class MiniQuadtrixHead(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        causal_mask = self.tril[:T, :T].bool()
        wei = wei.masked_fill(~causal_mask, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MiniQuadtrixMHA(nn.Module):
    def __init__(self, cfg, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MiniQuadtrixHead(cfg, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MiniQuadtrixFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class MiniQuadtrixBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = MiniQuadtrixMHA(cfg, cfg.n_head, head_size)
        self.ffwd = MiniQuadtrixFFN(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniQuadtrix(nn.Module):
    def __init__(self, cfg, vocab_size, device):
        super().__init__()
        self.cfg = cfg
        self.device_name = device
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[MiniQuadtrixBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device_name))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    args = parse_args()
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    start = time.time()

    log(f"{'mini-quadtrix-bpe':^{W}}")
    blank()
    row("Device", device)
    row("PyTorch", torch.__version__)

    text = args.data.read_text(encoding="utf-8")
    tokenizer, tokenizer_status = load_or_train_tokenizer(args, text)
    vocab_size = int(tokenizer.vocab_size)
    encoded_data = tokenizer.encode(text)
    data = torch.tensor(encoded_data, dtype=torch.long)

    if len(data) <= args.block_size + 1:
        raise ValueError("Dataset is too small for the configured block size")

    n = int(args.train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    if len(train_data) <= args.block_size + 1 or len(val_data) <= args.block_size + 1:
        raise ValueError("Train/validation split is too small for the configured block size")

    def get_batch(split):
        data_split = train_data if split == "train" else val_data
        ix = torch.randint(len(data_split) - args.block_size, (args.batch_size,))
        x = torch.stack([data_split[i : i + args.block_size] for i in ix])
        y = torch.stack([data_split[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = MiniQuadtrix(args, vocab_size, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    header("CONFIG")
    row("Seed", args.seed)
    row("Batch size", args.batch_size)
    row("Block size", args.block_size)
    row("Learning rate", args.learning_rate)
    row("Layers", args.n_layer)
    row("Heads", args.n_head)
    row("Embedding dim", args.n_embd)
    row("Dropout", args.dropout)
    row("Parameters", f"{n_params:,}")
    row("Tokenizer", tokenizer_status)
    row("Vocab size", vocab_size)
    row("Train tokens", f"{len(train_data):,}")
    row("Val tokens", f"{len(val_data):,}")
    row("Data file", str(args.data))
    row("Tokenizer file", str(args.tokenizer))

    header("TRAINING", f"{args.max_iters:,} steps | eval every {args.eval_interval} | checkpoint on improvement")
    blank()

    best_val_loss = float("inf")
    train_start = time.time()
    prev_loss = None

    for iter_num in range(args.max_iters):
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - train_start

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            loss_change = ""
            if prev_loss is not None:
                delta = losses["train"].item() - prev_loss
                loss_change = f"({delta:+.2f}z)"
            prev_loss = losses["train"].item()

            tokens_per_sec = (
                (iter_num + 1) * args.batch_size * args.block_size / elapsed
                if elapsed > 0
                else 0
            )

            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": vars(args),
                        "vocab_size": vocab_size,
                        "tokenizer": str(args.tokenizer),
                    },
                    args.checkpoint,
                )

            log(
                f"step {iter_num:>4}/{args.max_iters:<5} | "
                f"loss {losses['train']:.6f} {loss_change:<8} | "
                f"norm {total_norm:.4f} | "
                f"lr {args.learning_rate:.2e} | "
                f"{elapsed*1000:.2f} ms | "
                f"{int(tokens_per_sec)} tok/s"
            )
            sys.stdout.flush()

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - train_start
    blank()
    rule()
    row("Duration", f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
    row("Best val loss", f"{best_val_loss:.4f}", "", TICK)
    row("Checkpoint", str(args.checkpoint), "", TICK)
    rule()

    if args.no_chat:
        return

    blank()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    success(f"Restored {args.checkpoint} | val loss {best_val_loss:.4f}")

    header("INFERENCE", "quit / exit / q -> end session")
    blank()

    try:
        while True:
            prompt = input(f"  user  {ARROW} ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                blank()
                success("Session ended.")
                break
            if not prompt:
                continue

            encoded_prompt = tokenizer.encode(prompt)
            context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
            output_ids = model.generate(context, max_new_tokens=args.generate_tokens)
            new_tokens = output_ids[0][len(encoded_prompt) :].tolist()
            response = tokenizer.decode(new_tokens).strip()

            blank()
            log(f"  Model {ARROW} {response}")
            blank()
    except KeyboardInterrupt:
        blank()
        success("Interrupted.")

    wall_clock = time.time() - start
    blank()
    rule()
    row("Training", f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
    row("Total", f"{int(wall_clock // 60)}m {int(wall_clock % 60):02d}s", "", TICK)
    rule()
    blank()
    log(DOUBLE)


if __name__ == "__main__":
    main()
