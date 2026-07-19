#!/usr/bin/env bash
# GPTQ W4A16 quantization. Usage: bash 03_quantize_final_model_4bit.sh /path/to/final_model [sft.jsonl]

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
FINAL_MODEL_DIR="${1:-${FINAL_MODEL_DIR:-}}"
CALIBRATION_DATA="${2:-${CALIBRATION_DATA:-}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_DIR:-${RUN_ROOT}/runs/qwen3_6_27b_cpt_sft_gptq_w4a16_${RUN_ID}}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
CALIBRATION_MAX_SEQ_LEN="${CALIBRATION_MAX_SEQ_LEN:-2048}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-3900MB}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

require_file() {
  [[ -f "$1" ]] || {
    echo "Required input file not found: $1" >&2
    exit 1
  }
}

main() {
  [[ -n "$FINAL_MODEL_DIR" && -n "$CALIBRATION_DATA" ]] || {
    echo "Usage: $0 /path/to/final_model /path/to/sft.jsonl" >&2
    exit 1
  }
  require_command python
  require_file "${FINAL_MODEL_DIR}/config.json"
  require_file "$CALIBRATION_DATA"
  [[ ! -e "$QUANTIZED_MODEL_DIR" ]] || {
    echo "Quantized model output already exists: $QUANTIZED_MODEL_DIR" >&2
    exit 1
  }

  python - "$FINAL_MODEL_DIR" "$CALIBRATION_DATA" "$QUANTIZED_MODEL_DIR" \
    "$CALIBRATION_SAMPLES" "$CALIBRATION_MAX_SEQ_LEN" "$MAX_SHARD_SIZE" <<'PY'
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = Path(sys.argv[1])
calibration_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
sample_count = int(sys.argv[4])
max_seq_length = int(sys.argv[5])
max_shard_size = sys.argv[6]

records = []
with calibration_path.open("r", encoding="utf-8-sig") as file:
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

selected = random.Random(42).sample(records, k=min(sample_count, len(records)))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    model_path,
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
    max_seq_length=max_seq_length,
)

output_path.mkdir(parents=True, exist_ok=False)
model.save_pretrained(output_path, save_compressed=True, max_shard_size=max_shard_size)
tokenizer.save_pretrained(output_path)
print(f"4-bit GPTQ model saved to: {output_path}")
PY
}

main "$@"
