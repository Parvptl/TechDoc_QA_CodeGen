"""Prepare deploy-time CodeT5 model directory.

Strategy:
1) If a fine-tuned model already exists at target, keep it.
2) Else, optionally copy from a local source directory.
3) Else, download base CodeT5 model/tokenizer into target.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="models/codet5_finetuned")
    parser.add_argument("--source", default="models/codet5_finetuned_train_smoke")
    parser.add_argument("--base-model", default="Salesforce/codet5-small")
    args = parser.parse_args()

    target = Path(args.target)
    source = Path(args.source)

    # Keep existing model if present.
    if target.exists() and any(target.iterdir()):
        print(f"[INFO] CodeT5 target exists: {target}")
        return 0

    # Copy from local source if available.
    if source.exists() and any(source.iterdir()):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        print(f"[INFO] Copied CodeT5 model from {source} -> {target}")
        return 0

    # Download base model to target (deploy-safe fallback).
    print(f"[INFO] Downloading base CodeT5 model ({args.base_model}) to {target}")
    from transformers import RobertaTokenizer, T5ForConditionalGeneration

    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target)
    model.save_pretrained(target)
    print(f"[INFO] Saved base CodeT5 model to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

