#!/usr/bin/env python3
"""Dump reference logits from Hugging Face transformers for `cpuforge parity`.

Usage:
    python scripts/parity.py MODEL_DIR REF_DIR ["optional prompt text"]
    cpuforge parity --model MODEL_DIR --ref REF_DIR

Needs `torch` and `transformers` (any machine; CPU is fine). The model is run in
float32 so bf16 checkpoints are compared on equal terms with cpuforge, which
also computes in f32.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. In 1969, humans first walked on the Moon, and"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_dir, ref_dir = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PROMPT

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(ids).logits[0]  # [tokens, vocab]

    os.makedirs(ref_dir, exist_ok=True)
    with open(os.path.join(ref_dir, "tokens.txt"), "w") as f:
        f.write(" ".join(str(t) for t in ids[0].tolist()))
    logits.numpy().astype("<f4").tofile(os.path.join(ref_dir, "logits.bin"))
    print(f"wrote {ids.shape[1]} tokens x {logits.shape[1]} vocab to {ref_dir}")


if __name__ == "__main__":
    main()
