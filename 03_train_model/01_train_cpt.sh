#!/usr/bin/env bash
# Train a CPT LoRA adapter. Usage: bash 01_train_cpt.sh /path/to/cpt.jsonl

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
CPT_DATA="${1:-${CPT_DATA:-}}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${RUN_ROOT}/models/Qwen3.6-27B}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CPT_LORA_DIR="${CPT_LORA_DIR:-${RUN_DIR}/cpt_lora_${RUN_ID}}"

# Defaults for one RTX PRO 6000 96GB GPU.
CUTOFF_LEN="${CUTOFF_LEN:-6144}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
CPT_EPOCHS="${CPT_EPOCHS:-1.0}"
CPT_LEARNING_RATE="${CPT_LEARNING_RATE:-5e-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command not found: $1" >&2; exit 1; }
}

require_file() {
  [[ -f "$1" ]] || { echo "Required input file not found: $1" >&2; exit 1; }
}

usage() {
  echo "Usage: $0 /path/to/cpt.jsonl" >&2
}

main() {
  [[ -n "$CPT_DATA" ]] || { usage; exit 1; }
  require_command hf
  require_command llamafactory-cli
  require_file "$CPT_DATA"
  [[ ! -e "$CPT_LORA_DIR" ]] || { echo "Output already exists: $CPT_LORA_DIR" >&2; exit 1; }

  CPT_DATA="$(cd "$(dirname "$CPT_DATA")" && pwd -P)/$(basename "$CPT_DATA")"
  local dataset_dir
  dataset_dir="$(mktemp -d "${TMPDIR:-/tmp}/llamafactory_cpt_XXXXXX")"
  trap 'rm -rf -- "$dataset_dir"' EXIT
  mkdir -p "$BASE_MODEL_DIR" "$RUN_DIR"

  if [[ ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
    echo "Downloading ${MODEL_ID} to ${BASE_MODEL_DIR} ..."
    hf download "$MODEL_ID" --local-dir "$BASE_MODEL_DIR"
  fi

  cat > "${dataset_dir}/dataset_info.json" <<EOF
{
  "cpt": {
    "file_name": "${CPT_DATA}",
    "columns": {"prompt": "text"}
  }
}
EOF

  llamafactory-cli train \
    --stage pt --do_train true --model_name_or_path "$BASE_MODEL_DIR" \
    --dataset cpt --dataset_dir "$dataset_dir" --template empty \
    --finetuning_type lora --lora_target all --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --cutoff_len "$CUTOFF_LEN" --packing true --learning_rate "$CPT_LEARNING_RATE" \
    --num_train_epochs "$CPT_EPOCHS" --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing true --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" --bf16 true --flash_attn auto \
    --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 5 \
    --save_strategy steps --save_steps 500 --save_total_limit 2 --plot_loss true \
    --output_dir "$CPT_LORA_DIR" --report_to none

  rm -rf -- "$dataset_dir"
  trap - EXIT
  echo "CPT LoRA training completed: ${CPT_LORA_DIR}"
}

main "$@"
