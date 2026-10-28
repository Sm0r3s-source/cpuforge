# cpuforge

Finetune (and train small) Llama-style language models on a CPU, in Rust. No GPU, no PyTorch.

Supports the Llama family layout (RMSNorm, RoPE, grouped-query attention, SwiGLU): Llama (including 3.x rope scaling),
SmolLM, Mistral-style and Qwen3 models. Hand-written forward and backward passes, LoRA or full finetuning,
resumable checkpoints. Can run as the CPU training engine behind [Brewery](docs/brewery.md).

```
cargo build --release
./target/release/cpuforge demo          # self-contained tour, no downloads (run from this directory)
```

## Status: what is verified and what is not

| | |
|---|---|
| Every op's backward pass (RMSNorm, RoPE, SwiGLU, causal GQA attention, cross-entropy, LoRA) | checked against finite differences in f64 |
| Whole-model gradients (embedding to loss, tied and untied, LoRA and full) | checked against finite differences; deliberately broken backward code makes the tests fail |
| Memory-saving checkpoint mode | gives the same gradients as keeping all activations |
| Interrupt and resume | resumed run matches an uninterrupted run |
| Hugging Face directory loading (single file, sharded, bf16/f16), adapter files | round-trip tested |
| End-to-end CLI: init, train (full and LoRA), resume, generate, merge | tested on the real binary |
| Qwen3 (per-head q/k RMSNorm, explicit head_dim) | backward checked against finite differences; a deliberately wrong q-norm backward fails the tests |
| Llama 3.x rope scaling | frequency rule unit-tested and used in the whole-model tests; the formula was written from the Hugging Face definition and **not** compared against it |
| Brewery worker (`cpuforge worker`): status.json, metrics.jsonl, STOP file / SIGTERM, resume, PEFT adapter output | tested on the real binary, and from Brewery's own `brewery train` |
| `cpuforge pull` (model download) | tested against a fake Hugging Face server (partial files, gated models, hostile file names); never run against the real site |
| **Matching PyTorch/transformers on a real model** | **not yet run.** The sandbox this was built in cannot reach huggingface.co. Run the parity check below on your machine. |

## The app

One file, no installer: double-click `cpuforge` (`cpuforge.exe` on Windows) and the app opens in your browser
(it serves a page on 127.0.0.1:7860 and launches your default browser; the small console window is the server, close it to quit).
From a terminal: `cpuforge ui [--port N] [--no-open]`. Run it from a folder that holds `models/` and `data/` (created if missing).

## Releases and self-update

Every push to `main` runs `.github/workflows/release.yml`: tests, then Linux, Windows and macOS programs published as release
`build-N` (the program's version is `0.N`). Each asset has a `.sha256` file that is checked before installing.

- `cpuforge update [--check]` installs the newest release over the running program. The UI header has the same as a button
  (Check for updates, then Install, then it restarts itself with the same arguments).
- `cpuforge ui --auto-update` checks every 6 hours and installs when no job is running. It is ignored on local dev builds.
- Repo is `Sm0r3s-source/cpuforge` unless `CPUFORGE_REPO=owner/repo` (or `--repo`) says otherwise. A private repo needs `CPUFORGE_TOKEN`.
- Build on Windows for Windows; a headless Linux box needs the Linux asset, which the workflow builds.

### Headless Linux box

```
curl -fsSL https://raw.githubusercontent.com/Sm0r3s-source/cpuforge/main/scripts/install-sbook.sh | bash
```

Installs `~/.local/bin/cpuforge`, a `cpuforge.service` (systemd, restarts on failure, 6 GB memory cap) serving the app on the
Tailscale address on port 7860, and `--auto-update`. Models and data live in `/data/cpuforge`. There is no login: it is reachable
wherever the bind address is, so keep it on Tailscale or on 127.0.0.1 plus `ssh -L 7860:127.0.0.1:7860`.

A local web app (nothing leaves your computer): pick a model and data, press Train, watch the loss curve and log live,
generate text with or without an adapter, merge adapters, create a from-scratch model, paste training text.
Run it from the folder that holds your `models/` and `data/`. Every button runs the matching CLI command as a child
process, one job at a time, with Cancel. It only listens on 127.0.0.1, refuses foreign `Host`/`Origin` headers and
non-JSON POSTs, and only accepts relative paths without `..`.

## Quick start with a real model

```
# 1. get a model (any Llama-style checkpoint with tokenizer.json), e.g.
huggingface-cli download HuggingFaceTB/SmolLM-135M --local-dir models/SmolLM-135M

# 2. check that cpuforge reproduces PyTorch's logits (needs torch + transformers once)
python scripts/parity.py models/SmolLM-135M /tmp/ref
./target/release/cpuforge parity --model models/SmolLM-135M --ref /tmp/ref      # expect PASS

# 3. finetune with LoRA
./target/release/cpuforge train --config examples/smollm-lora.toml

# 4. try it, or bake the adapter into a plain model directory
./target/release/cpuforge generate --model models/SmolLM-135M --adapter runs/smollm-demo/adapter.safetensors --prompt "Q: What is Rust?\nA:"
./target/release/cpuforge merge --model models/SmolLM-135M --adapter runs/smollm-demo/adapter.safetensors --out models/SmolLM-135M-tuned
```

Training data is either plain text (cut into windows of `seq_len`) or JSON lines:
`{"text": "..."}` (loss on every token) or `{"prompt": "...", "completion": "..."}` (loss on the completion only).
Add `--resume` to continue an interrupted run from `out_dir/state.safetensors`.

Training from scratch: `cpuforge init --out models/mine --preset small`, then a config with `tokenizer = "bytes"` and `mode = "full"`.

## Speed and memory

Measured with `cpuforge bench-train` on a **2-core cloud VM** (random SmolLM-135M-shaped model, 134.5M parameters,
512-token sequences, f32):

| | tokens/s | s per 512-token step |
|---|---|---|
| prompt processing (forward only) | 353 | 1.5 |
| LoRA r=16 on all projections | 122 | 4.2 |
| LoRA + gradient checkpointing | 92 | 5.6 |
| full finetune | 89 | 5.8 |

Peak memory 1.7 GB. Roughly: 1M training tokens in about 2.3 hours on 2 cores for a 135M model, and it should scale with
cores. Matmul runs at 120-150 GFLOP/s here (`cpuforge bench`). These are not numbers for your machine: run the benchmark.

Qwen3-0.6B-shaped model (596M params, random fp16 weights in the real Qwen3 tensor layout, LoRA rank 16 on all projections, checkpoint mode, 256-token sequences) on the same 2 cores: about 22 tokens/s, 3.5 GB peak RAM.

## Layout

| crate | purpose |
|---|---|
| `cf-core` | tensor, `gemm`-backed matmul, RMSNorm / RoPE / SwiGLU / attention / cross-entropy with backward, grad-check helpers |
| `cf-model` | config, LoRA linear layer, transformer block, full model forward/backward, parameter visitor |
| `cf-io` | safetensors (mmap) reader/writer, Hugging Face directories, adapter files |
| `cf-train` | AdamW, LR schedule, tokenizers and datasets, trainer, resumable state |
| `cf-cli` | the `cpuforge` binary |

Conventions: every `*_backward` accumulates (`+=`) into gradient buffers. An empty gradient buffer means "frozen", so a
frozen base weight costs no weight-gradient matmul.

`.cargo/config.toml` builds with `target-cpu=native` for speed; delete it if you need a binary for other machines.

## Not supported yet

- Models with attention biases (Qwen2), other `rope_scaling` types (yarn, linear, ...), Gemma, MoE: the loader refuses them with a clear error rather than silently computing wrong numbers.
- KV cache: `generate` recomputes the sequence per token, fine for checking a finetune, slow for long outputs.
- Quantized base weights (the main way to fit 0.5B+ models in little RAM), GGUF loading.
- Batching several sequences into one forward pass; logits are materialised as `[tokens, vocab]`, which is large for 150k-token vocabularies.
- Gradient accumulation averages per-sequence mean losses rather than weighting by token count (recent Hugging Face trainers weight by token count, so loss values and gradient scale differ slightly on uneven batches).
- DPO and preference data.
