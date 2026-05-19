# llm

Train, sample, and benchmark a compact GPT-style language model from your own text.

`qchat` is a local LLM pipeline with a custom byte-level BPE tokenizer, a PyTorch decoder-only Transformer, checkpointed training, standalone inference, and repeatable benchmarks. It is designed as a usable experimentation stack: put text in, train a model, generate from it, measure it, iterate.

```text
data/input.txt -> tokenizer/tokenizer.json -> train.py -> mini-quadtrix-bpe.pt -> inference.py
```

## Features

- **End-to-end local workflow**: dataset, tokenizer, trainer, checkpoint, inference, benchmarks.
- **Custom BPE tokenizer**: byte-level encoding with no unknown-token failure mode.
- **GPT-style model**: causal self-attention, MLP blocks, layer norm, learned positions, LM head.
- **Checkpoint-aware inference**: model config and tokenizer path are restored from the saved checkpoint.
- **Interactive and one-shot generation**: use it as a prompt runner or a terminal chat loop.
- **Benchmark runner**: measure tokenization, batch creation, forward pass, generation, memory, and system metadata.
- **Multiple execution paths**: PyTorch CPU/CUDA mainline, plus older C and DirectML/iGPU experiments.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch
```

For optional benchmark memory reporting:

```powershell
pip install psutil
```

## Quick Start

Run a fast smoke test:

```powershell
python train.py --quick --no-chat
```

Train from `data/input.txt`:

```powershell
python train.py --no-chat
```

Generate from the saved checkpoint:

```powershell
python inference.py --prompt "Once upon a time"
```

Open interactive generation:

```powershell
python inference.py
```

Run benchmarks:

```powershell
python benchmarks/benchmark.py
```

## Standard Workflow

### 1. Add Data

Place your corpus here:

```text
data/input.txt
```

Or point the trainer at another file:

```powershell
python train.py --data data\my_corpus.txt --no-chat
```

### 2. Train

```powershell
python train.py --no-chat
```

By default, training writes:

```text
mini-quadtrix-bpe.pt
```

The checkpoint contains the model weights, config, vocab size, and tokenizer path.

### 3. Generate

```powershell
python inference.py `
  --checkpoint mini-quadtrix-bpe.pt `
  --prompt "Write a short answer about"
```

Control sampling:

```powershell
python inference.py `
  --prompt "The system should" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 50
```

### 4. Benchmark

```powershell
python benchmarks/benchmark.py `
  --checkpoint mini-quadtrix-bpe.pt `
  --data data\input.txt `
  --tokenizer tokenizer\tokenizer.json
```

Results are written under:

```text
benchmarks/results/
```

## Common Commands

Train a tiny verification model:

```powershell
python train.py --quick --no-chat
```

Retrain the tokenizer and model:

```powershell
python train.py --retrain-tokenizer --no-chat
```

Use a custom checkpoint and tokenizer:

```powershell
python train.py `
  --checkpoint checkpoints\run-a.pt `
  --tokenizer tokenizer\run-a.json `
  --retrain-tokenizer `
  --no-chat
```

Run inference on CPU:

```powershell
python inference.py --device cpu --prompt "Hello"
```

Run inference on CUDA:

```powershell
python inference.py --device cuda --prompt "Hello"
```

Use GPT-2 tokenization mode for older compatible checkpoints:

```powershell
python benchmarks/benchmark.py --tokenizer-kind gpt2
```

## Configuration

Main training flags:

| Flag | Default | Description |
|---|---:|---|
| `--data` | `data/input.txt` | Training text |
| `--tokenizer` | `tokenizer/tokenizer.json` | BPE tokenizer file |
| `--checkpoint` | `mini-quadtrix-bpe.pt` | Output checkpoint |
| `--vocab-size` | `8192` | Target tokenizer vocabulary size |
| `--tokenizer-train-chars` | `5000000` | Text chars used to train BPE |
| `--retrain-tokenizer` | off | Rebuild tokenizer even if file exists |
| `--train-split` | `0.9` | Train/validation split |
| `--seed` | `1337` | Random seed |
| `--batch-size` | `2` | Sequences per training step |
| `--block-size` | `8192` | Context length |
| `--max-iters` | `10000` | Training steps |
| `--eval-interval` | `10` | Validation frequency |
| `--eval-iters` | `20` | Validation batches per estimate |
| `--learning-rate` | `3e-4` | AdamW learning rate |
| `--n-embd` | `6144` | Embedding width |
| `--n-head` | `48` | Attention heads |
| `--n-layer` | `48` | Transformer layers |
| `--dropout` | `0.0` | Dropout |
| `--generate-tokens` | `200` | Tokens generated after training chat starts |
| `--no-chat` | off | Exit after training |
| `--quick` | off | Use a tiny smoke-test config |

Inference flags:

| Flag | Default | Description |
|---|---:|---|
| `--checkpoint` | `mini-quadtrix-bpe.pt` | Model checkpoint |
| `--tokenizer` | checkpoint tokenizer | Override tokenizer path |
| `--prompt` | interactive mode | One-shot prompt |
| `--max-new-tokens` | `200` | Generation length |
| `--temperature` | `1.0` | Sampling randomness |
| `--top-k` | none | Restrict sampling to top-k tokens |
| `--device` | auto | `cpu`, `cuda`, or another PyTorch device |

## Model Size Notes

The default config is large for many local machines:

```text
n_layer=48
n_head=48
n_embd=6144
block_size=8192
```

For practical local iteration, start smaller:

```powershell
python train.py `
  --batch-size 8 `
  --block-size 256 `
  --n-embd 384 `
  --n-head 6 `
  --n-layer 6 `
  --max-iters 2000 `
  --eval-interval 100 `
  --no-chat
```

Then scale one dimension at a time.

## Repository Layout

```text
data/
  input.txt                 default training corpus
  data_set.py               dataset helper

tokenizer/
  bpe.py                    byte-level BPE tokenizer
  tokenizer.json            saved tokenizer
  __init__.py

train.py                    main model and training loop
inference.py                checkpoint inference runner

benchmarks/
  benchmark.py              benchmark suite
  README.md                 benchmark notes
  results/                  generated results

engine/
  main.py                   older/reference training path
  inference.py              older/reference inference path
  engine.c                  C inference experiment
  export_weights.py         PyTorch weight export helper
  fine-tune/                fine-tuning experiments

src/
  directml/                 experimental iGPU/DirectML path
  large_gpu/                older large-GPU experiment
  scratch/                  small experiments

assets/                     run screenshots and artifacts
tools/                      utility scripts
.github/workflows/          CI, CodeQL, and benchmark workflows
.vscode/                    local tasks and debug configs
```

## Benchmark Outputs

The benchmark runner records:

- tokenizer speed
- batch creation speed
- forward latency
- generation latency
- optional training-step latency
- memory usage when available
- system metadata

Tokenizer selection:

```powershell
python benchmarks/benchmark.py --tokenizer-kind auto
python benchmarks/benchmark.py --tokenizer-kind bpe
python benchmarks/benchmark.py --tokenizer-kind gpt2
```

`auto` chooses GPT-2 tokenization for older `50257` vocab checkpoints and custom BPE otherwise.

## Experiment Log

| # | Time | Val BPB / Loss | Core | Description | Date | Contributor |
|---:|---:|---:|---:|---|---|---|
| 0 | 39.4 min | 1.3145 | 0.82M | CPU baseline, small data, fragmented output | 2026 | @Eamon2009 |
| 1 | 61.3 min | 0.7176 | 10.82M | Colab large-scale run, coherent paragraphs, stronger convergence | 2026 | @Eamon2009 |
| 2 | 6.1 min | 0.9250 | 1.99M | T4 optimized run, fast training, stable learning, basic coherence | 2026 | @Eamon2009 |
| 3 | 76.2 min | 1.6371 | ~0.82M | C++ extended CPU training, 3000 iterations | 2026 | @Eamon2009 |

## Troubleshooting

**Checkpoint not found**

Run training first:

```powershell
python train.py --quick --no-chat
```

**Tokenizer vocab size does not match checkpoint**

Use the tokenizer that was saved with the checkpoint, or pass it explicitly:

```powershell
python inference.py --checkpoint path\model.pt --tokenizer path\tokenizer.json
```

**Dataset is too small for the configured block size**

Reduce context length:

```powershell
python train.py --block-size 128 --no-chat
```

**Out of memory**

Reduce `--batch-size`, `--block-size`, `--n-embd`, or `--n-layer`.

**Repetitive generations**

Try a higher `--temperature`, use `--top-k`, train longer, or improve the corpus quality.

## License
 MIT
