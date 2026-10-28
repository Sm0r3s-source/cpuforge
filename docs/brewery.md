# cpuforge as a CPU engine for Brewery

[Brewery](https://github.com/empero-org/brewery-ai) is an agent that guides you through fine-tuning a model: data,
hyperparameters, training, export. It trains with PyTorch, which in practice means a GPU. cpuforge can take over the
training step on machines without one. Brewery keeps everything else.

```
Brewery agent ──► job.yaml (runtime.backend: cpuforge) ──► brewery train
                                                              │ tokenizes with the model's chat template
                                                              ▼
                                          data/*.tok.jsonl + cpuforge.toml
                                                              │
                                                              ▼
                                      cpuforge worker ──► status.json, metrics.jsonl, final/ (PEFT adapter)
```

Brewery side: see `docs/cpuforge.md` in the Brewery change (`brewery-cpuforge.patch`). cpuforge side, this repo:

## `cpuforge worker --config cpuforge.toml [--resume]`

Runs one job and follows Brewery's worker contract, in the folder that holds the config:

| file | content |
|---|---|
| `status.json` | state (`preparing`, `training`, `saving`, `completed`, `failed`, `stopped`), step, max_steps, loss history, params, ETA, and on failure `error`, `error_kind`, `hint` |
| `metrics.jsonl` | one line per step (`step`, `loss`, `learning_rate`, `grad_norm`, `epoch`, `tokens_per_sec`) and per evaluation (`eval_loss`) |
| `STOP` | create this file, or send SIGTERM / Ctrl-C: the worker stops promptly (even in the middle of a step), saves `state.safetensors` as of the last completed step, and exits with state `stopped` |
| `state.safetensors` | resumable state; `--resume` continues from the exact step |
| `final/` | LoRA: `adapter_config.json` + `adapter_model.safetensors` in the Hugging Face PEFT layout (loads with `PeftModel.from_pretrained`); full: `model.safetensors` + `config.json`; plus the tokenizer files found next to the model |
| `results.json` | metrics and whether the run stopped early |

Config (TOML): `[run]` job_id, base_model, objective; `[model]` `hf = "org/name"` (downloaded) or `path = "dir"`;
`[data]` pre-tokenized JSONL with `input_ids` and `labels` (`-100` = not learned), seq_len; `[lora]` rank, alpha, targets;
`[train]` steps, batch, lr, warmup_steps, weight_decay, clip, memory (`full` | `checkpoint`), eval_every, save_every.

## `cpuforge pull --repo org/name [--out DIR] [--revision main]`

Downloads the safetensors weights, config and tokenizer files (not `.bin`, ONNX, GGUF). Needs no Python. Gated models
need `HF_TOKEN`. Complete files are skipped, so an interrupted pull continues. Default folder:
`~/.cache/cpuforge/models/org__name` (`CPUFORGE_HOME` changes the root).

## `generate` and `merge` with PEFT adapters

`cpuforge generate --adapter DIR` and `cpuforge merge --adapter DIR` accept both cpuforge's own adapter files and a PEFT
folder, so an adapter trained by Brewery's GPU worker can be tried on a CPU box and the reverse.

## Honest limits

- Overnight scale means about 0.6B parameters and below (Qwen3-0.6B, Llama-3.2-1B with patience). Measured on a 2-core
  cloud VM with AVX2: a Qwen3-0.6B-shaped model (596M parameters, random weights, LoRA rank 16 on all projections,
  `memory = "checkpoint"`, 256-token sequences) trains at about 22 tokens/s and peaks at 3.5 GB of RAM, so ten hours
  cover roughly 800k tokens. A 135M model runs about 5x faster. Your CPU will differ: run a few steps first.
- Not compared token for token with PyTorch/PEFT on a real checkpoint. Gradients and every op are verified
  mathematically, but the first run on a real model should be checked with `brewery test`.
- Supported: Llama, Mistral, Qwen3 architectures; SFT and CPT-style data. Not: DPO, Gemma, MoE, image models, QLoRA.
