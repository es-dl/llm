
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Change these paths for the usual workflow.
CHECKPOINT_PATH = ROOT / "mini-quadtrix-bpe.pt"
DATA_PATH = ROOT / "data" / "input.txt"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = ROOT / "benchmarks" / "results"

# auto: use BPE when TOKENIZER_PATH exists, otherwise GPT-2/tiktoken.
# bpe: force the custom tokenizer.
# gpt2: force tiktoken GPT-2.
TOKENIZER_KIND = "auto"


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.0


@dataclass
class BenchRow:
    suite: str
    name: str
    checkpoint: str
    batch_size: int = 0
    sequence_length: int = 0
    tokens: int = 0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    std_ms: float = 0.0
    tokens_per_sec: float = 0.0
    samples: int = 0
    loss: float | None = None
    memory_mb: float | None = None
    notes: str = ""


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_ms(samples: list[float]) -> dict[str, float]:
    return {
        "avg_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p90_ms": percentile(samples, 0.90),
        "p95_ms": percentile(samples, 0.95),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_samples(
    device: torch.device,
    fn: Callable[[], object],
    runs: int,
    warmup: int,
) -> tuple[list[float], object]:
    last = None
    for _ in range(warmup):
        last = fn()
    sync(device)

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        last = fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, last


def process_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        return None


def cuda_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint must be a state_dict or a dict containing a 'model' state_dict")


def infer_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    token_shape = state_dict["token_embedding_table.weight"].shape
    pos_shape = state_dict["position_embedding_table.weight"].shape
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    head_ids = {
        int(key.split(".")[4])
        for key in state_dict
        if key.startswith("blocks.0.sa.heads.") and key.split(".")[4].isdigit()
    }

    return ModelConfig(
        vocab_size=int(token_shape[0]),
        n_embd=int(token_shape[1]),
        block_size=int(pos_shape[0]),
        n_layer=max(layer_ids) + 1,
        n_head=max(head_ids) + 1,
        dropout=0.0,
    )


class Head(nn.Module):
    def __init__(self, cfg: ModelConfig, head_size: int):
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
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(head_size * cfg.n_head, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTBenchmarkModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
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
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
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


class GPT2TokenizerAdapter:
    def __init__(self):
        import tiktoken

        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


def load_tokenizer(kind: str, tokenizer_path: Path, cfg: ModelConfig):
    if kind == "auto":
        kind = "gpt2" if cfg.vocab_size == 50257 else "bpe"

    if kind == "bpe":
        from tokenizer import BytePairTokenizer

        return BytePairTokenizer.load(tokenizer_path), "bpe"
    if kind == "gpt2":
        return GPT2TokenizerAdapter(), "gpt2"
    raise ValueError("tokenizer kind must be one of: auto, bpe, gpt2")


class CheckpointBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        self.state_dict = unwrap_state_dict(checkpoint)
        self.cfg = infer_config(self.state_dict)
        self.tokenizer, self.tokenizer_kind = load_tokenizer(args.tokenizer_kind, args.tokenizer, self.cfg)
        self.model = GPTBenchmarkModel(self.cfg, self.device).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.eval()
        self.rows: list[BenchRow] = []

    def record(self, row: BenchRow) -> None:
        self.rows.append(row)
        print(
            f"{row.suite:<12} {row.name:<24} "
            f"avg={row.avg_ms:9.3f} ms  "
            f"p95={row.p95_ms:9.3f} ms  "
            f"tok/s={row.tokens_per_sec:10.1f}"
        )

    def run(self) -> dict:
        print("Quadtrix Checkpoint Benchmark")
        print(f"Checkpoint: {self.args.checkpoint}")
        print(f"Device: {self.device}")
        print(f"Tokenizer: {self.tokenizer_kind}")
        print(f"Model: {self.cfg}")
        print(f"Runs: {self.args.runs}, warmup: {self.args.warmup}")

        self.bench_tokenizer_and_data()
        self.bench_forward()
        self.bench_generation()
        if not self.args.skip_train_step:
            self.bench_training_step()
        return self.save()

    def bench_tokenizer_and_data(self) -> None:
        text = self.args.data.read_text(encoding="utf-8")
        if self.args.max_data_chars and len(text) > self.args.max_data_chars:
            text = text[: self.args.max_data_chars]

        samples, encoded = timed_samples(
            self.device,
            lambda: self.tokenizer.encode(text),
            self.args.runs,
            self.args.warmup,
        )
        stats = summarize_ms(samples)
        self.record(
            BenchRow(
                suite="data",
                name="tokenizer_encode",
                checkpoint=str(self.args.checkpoint),
                tokens=len(encoded),
                tokens_per_sec=len(encoded) / (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )

        tensor = torch.tensor(encoded, dtype=torch.long)
        seq_len = min(self.args.sequence_length or self.cfg.block_size, self.cfg.block_size, max(2, len(tensor) - 2))
        batch_size = min(self.args.batch_size, max(1, len(tensor) - seq_len - 1))

        def make_batch():
            ix = torch.randint(len(tensor) - seq_len - 1, (batch_size,))
            x = torch.stack([tensor[i : i + seq_len] for i in ix]).to(self.device)
            y = torch.stack([tensor[i + 1 : i + seq_len + 1] for i in ix]).to(self.device)
            return x, y

        samples, _ = timed_samples(self.device, make_batch, self.args.runs, self.args.warmup)
        stats = summarize_ms(samples)
        self.record(
            BenchRow(
                suite="data",
                name="batch_sample_to_device",
                checkpoint=str(self.args.checkpoint),
                batch_size=batch_size,
                sequence_length=seq_len,
                tokens=batch_size * seq_len,
                tokens_per_sec=(batch_size * seq_len) / (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )

    def bench_forward(self) -> None:
        cases = [
            (1, min(8, self.cfg.block_size)),
            (1, min(self.args.sequence_length or self.cfg.block_size, self.cfg.block_size)),
            (self.args.batch_size, min(self.args.sequence_length or self.cfg.block_size, self.cfg.block_size)),
        ]

        for batch_size, seq_len in cases:
            idx = torch.randint(self.cfg.vocab_size, (batch_size, seq_len), device=self.device)
            targets = torch.randint(self.cfg.vocab_size, (batch_size, seq_len), device=self.device)

            @torch.no_grad()
            def fn():
                return self.model(idx, targets)

            samples, last = timed_samples(self.device, fn, self.args.runs, self.args.warmup)
            stats = summarize_ms(samples)
            tokens = batch_size * seq_len
            self.record(
                BenchRow(
                    suite="forward",
                    name=f"batch{batch_size}_seq{seq_len}",
                    checkpoint=str(self.args.checkpoint),
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    tokens=tokens,
                    tokens_per_sec=tokens / (stats["avg_ms"] / 1000.0),
                    samples=len(samples),
                    loss=float(last[1].item()),
                    memory_mb=process_rss_mb(),
                    **stats,
                )
            )

    def bench_generation(self) -> None:
        prompts = [
            ("empty", ""),
            ("short", "The future of local AI is"),
            ("long", "Quadtrix is a compact transformer benchmark that measures " * 4),
        ]

        for label, prompt in prompts:
            encoded = self.tokenizer.encode(prompt) or [0]
            encoded = encoded[-self.cfg.block_size :]
            idx = torch.tensor([encoded], dtype=torch.long, device=self.device)

            @torch.no_grad()
            def fn():
                return self.model.generate(
                    idx,
                    self.args.generate_tokens,
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                )

            samples, _ = timed_samples(self.device, fn, self.args.runs, self.args.warmup)
            stats = summarize_ms(samples)
            self.record(
                BenchRow(
                    suite="generation",
                    name=label,
                    checkpoint=str(self.args.checkpoint),
                    batch_size=1,
                    sequence_length=len(encoded),
                    tokens=self.args.generate_tokens,
                    tokens_per_sec=self.args.generate_tokens / (stats["avg_ms"] / 1000.0),
                    samples=len(samples),
                    memory_mb=process_rss_mb(),
                    **stats,
                )
            )

    def bench_training_step(self) -> None:
        model = GPTBenchmarkModel(self.cfg, self.device).to(self.device)
        model.load_state_dict(self.state_dict, strict=False)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        seq_len = min(self.args.sequence_length or self.cfg.block_size, self.cfg.block_size)
        idx = torch.randint(self.cfg.vocab_size, (self.args.batch_size, seq_len), device=self.device)
        targets = torch.randint(self.cfg.vocab_size, (self.args.batch_size, seq_len), device=self.device)

        def fn():
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            return loss.detach()

        samples, loss = timed_samples(self.device, fn, self.args.train_steps, self.args.warmup)
        stats = summarize_ms(samples)
        tokens = self.args.batch_size * seq_len
        self.record(
            BenchRow(
                suite="training",
                name=f"adamw_step_b{self.args.batch_size}_s{seq_len}",
                checkpoint=str(self.args.checkpoint),
                batch_size=self.args.batch_size,
                sequence_length=seq_len,
                tokens=tokens,
                tokens_per_sec=tokens / (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                loss=float(loss.item()),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )

    def save(self) -> dict:
        self.args.out.mkdir(parents=True, exist_ok=True)
        n_params = sum(p.numel() for p in self.model.parameters())
        stamp = time.strftime("%Y%m%d_%H%M%S")
        result = {
            "schema_version": 2,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "system": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "device": str(self.device),
                "cuda": getattr(torch.version, "cuda", None),
                "rss_mb": process_rss_mb(),
                **cuda_memory(self.device),
            },
            "model": {
                **asdict(self.cfg),
                "parameters": n_params,
                "parameter_mb_fp32": n_params * 4 / (1024**2),
            },
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args).items()
            },
            "results": [asdict(row) for row in self.rows],
        }

        json_path = self.args.out / f"benchmark_{stamp}.json"
        csv_path = self.args.out / f"benchmark_{stamp}.csv"
        latest_json = self.args.out / "latest.json"
        latest_csv = self.args.out / "latest.csv"

        json_text = json.dumps(result, indent=2)
        json_path.write_text(json_text, encoding="utf-8")
        latest_json.write_text(json_text, encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.rows[0]).keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"Saved {json_path}")
        print(f"Saved {csv_path}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark any compatible Quadtrix .pt checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--tokenizer-kind", choices=["auto", "bpe", "gpt2"], default=TOKENIZER_KIND)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None, help="Example: cpu, cuda, cuda:0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--generate-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-data-chars", type=int, default=1_000_000)
    parser.add_argument("--skip-train-step", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.runs = 2
        args.warmup = 1
        args.train_steps = 1
        args.generate_tokens = 4
        args.sequence_length = min(args.sequence_length or 64, 64)
        args.max_data_chars = min(args.max_data_chars, 50_000)
        args.skip_train_step = True

    return args


def main() -> int:
    try:
        CheckpointBenchmark(parse_args()).run()
        return 0
    except ImportError as exc:
        print(f"Missing benchmark dependency: {exc}", file=sys.stderr)
        print("Install torch, plus tiktoken when using --tokenizer-kind gpt2.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
