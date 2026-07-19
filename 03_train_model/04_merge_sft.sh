#!/usr/bin/env bash
# Merge an SFT LoRA adapter. Usage: bash 04_merge_sft.sh /path/to/sft_lora /path/to/cpt_merged_model

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
SFT_LORA_DIR="${1:-${SFT_LORA_DIR:-}}"
BASE_MODEL_DIR="${2:-${CPT_MERGED_DIR:-}}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-${RUN_DIR}/qwen3_6_27b_cpt_sft_merged_${RUN_ID}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_SIZE="${EXPORT_SIZE:-4}"

usage() { echo "Usage: $0 /path/to/sft_lora /path/to/cpt_merged_model" >&2; }

main() {
  [[ -n "$SFT_LORA_DIR" && -n "$BASE_MODEL_DIR" ]] || { usage; exit 1; }
  command -v llamafactory-cli >/dev/null 2>&1 || { echo "Required command not found: llamafactory-cli" >&2; exit 1; }
  [[ -f "${BASE_MODEL_DIR}/config.json" ]] || { echo "CPT merged model not found: $BASE_MODEL_DIR" >&2; exit 1; }
  [[ -f "${SFT_LORA_DIR}/adapter_config.json" ]] || { echo "SFT LoRA adapter not found: $SFT_LORA_DIR" >&2; exit 1; }
  [[ ! -e "$FINAL_MODEL_DIR" ]] || { echo "Output already exists: $FINAL_MODEL_DIR" >&2; exit 1; }
  mkdir -p "$RUN_DIR"

  llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL_DIR" --adapter_name_or_path "$SFT_LORA_DIR" \
    --template qwen3 --finetuning_type lora --export_dir "$FINAL_MODEL_DIR" \
    --export_size "$EXPORT_SIZE" --export_device "$EXPORT_DEVICE" --export_legacy_format false

  echo "Final model merged: ${FINAL_MODEL_DIR}"
}

main "$@"
