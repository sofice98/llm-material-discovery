#!/usr/bin/env bash
# Merge a CPT LoRA adapter. Usage: bash 02_merge_cpt.sh /path/to/cpt_lora

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
CPT_LORA_DIR="${1:-${CPT_LORA_DIR:-}}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${RUN_ROOT}/models/Qwen3.6-27B}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CPT_MERGED_DIR="${CPT_MERGED_DIR:-${RUN_DIR}/cpt_merged_${RUN_ID}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_SIZE="${EXPORT_SIZE:-4}"

usage() { echo "Usage: $0 /path/to/cpt_lora" >&2; }

main() {
  [[ -n "$CPT_LORA_DIR" ]] || { usage; exit 1; }
  command -v llamafactory-cli >/dev/null 2>&1 || { echo "Required command not found: llamafactory-cli" >&2; exit 1; }
  [[ -f "${BASE_MODEL_DIR}/config.json" ]] || { echo "Base model not found: $BASE_MODEL_DIR" >&2; exit 1; }
  [[ -f "${CPT_LORA_DIR}/adapter_config.json" ]] || { echo "CPT LoRA adapter not found: $CPT_LORA_DIR" >&2; exit 1; }
  [[ ! -e "$CPT_MERGED_DIR" ]] || { echo "Output already exists: $CPT_MERGED_DIR" >&2; exit 1; }
  mkdir -p "$RUN_DIR"

  llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL_DIR" --adapter_name_or_path "$CPT_LORA_DIR" \
    --template qwen3 --finetuning_type lora --export_dir "$CPT_MERGED_DIR" \
    --export_size "$EXPORT_SIZE" --export_device "$EXPORT_DEVICE" --export_legacy_format false

  echo "CPT model merged: ${CPT_MERGED_DIR}"
}

main "$@"
