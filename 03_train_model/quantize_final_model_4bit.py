#!/usr/bin/env python3
"""Quantize a merged BF16 model to GPTQ W4A16 compressed-tensors."""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_model_dir", type=Path)
    parser.add_argument("calibration_data", type=Path)
    parser.add_argument("quantized_model_dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-max-seq-len", type=int, default=1024)
    parser.add_argument("--max-shard-size", default="3900MB")
    return parser.parse_args()


def load_calibration_records(path: Path) -> list[tuple[str, str, str]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            instruction = str(record.get("instruction", "")).strip()
            user_input = str(record.get("input", "")).strip()
            output = str(record.get("output", "")).strip()
            if instruction and output:
                records.append((instruction, user_input, output))
    if not records:
        raise RuntimeError("Calibration JSONL has no valid SFT records.")
    return records


def main() -> None:
    args = parse_args()
    records = load_calibration_records(args.calibration_data)
    selected = random.Random(42).sample(
        records, k=min(args.calibration_samples, len(records))
    )

    tokenizer = AutoTokenizer.from_pretrained(args.final_model_dir, trust_remote_code=True)
    texts = []
    for instruction, user_input, output in selected:
        user_content = instruction if not user_input else f"{instruction}\n\n{user_input}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.final_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(
        model=model,
        recipe=recipe,
        dataset=Dataset.from_dict({"text": texts}),
        num_calibration_samples=len(texts),
        max_seq_length=args.calibration_max_seq_len,
    )

    args.quantized_model_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        args.quantized_model_dir,
        save_compressed=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.quantized_model_dir)
    print(f"4-bit GPTQ model saved to: {args.quantized_model_dir}")


if __name__ == "__main__":
    main()
