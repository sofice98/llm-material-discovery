#!/usr/bin/env bash
# Train an SFT LoRA adapter. Usage: bash 03_train_sft.sh /path/to/sft.jsonl /path/to/cpt_merged_model

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
SFT_DATA="${1:-${SFT_DATA:-}}"
BASE_MODEL_DIR="${2:-${CPT_MERGED_DIR:-}}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SFT_LORA_DIR="${SFT_LORA_DIR:-${RUN_DIR}/sft_lora_${RUN_ID}}"

# Defaults for one RTX PRO 6000 96GB GPU.
CUTOFF_LEN="${CUTOFF_LEN:-6144}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SFT_EPOCHS="${SFT_EPOCHS:-3.0}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
SFT_VAL_SIZE="${SFT_VAL_SIZE:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-100}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() { echo "Usage: $0 /path/to/sft.jsonl /path/to/cpt_merged_model" >&2; }

main() {
  [[ -n "$SFT_DATA" && -n "$BASE_MODEL_DIR" ]] || { usage; exit 1; }
  command -v llamafactory-cli >/dev/null 2>&1 || { echo "Required command not found: llamafactory-cli" >&2; exit 1; }
  [[ -f "$SFT_DATA" ]] || { echo "Required input file not found: $SFT_DATA" >&2; exit 1; }
  [[ -f "${BASE_MODEL_DIR}/config.json" ]] || { echo "CPT merged model not found: $BASE_MODEL_DIR" >&2; exit 1; }
  [[ ! -e "$SFT_LORA_DIR" ]] || { echo "Output already exists: $SFT_LORA_DIR" >&2; exit 1; }

  SFT_DATA="$(cd "$(dirname "$SFT_DATA")" && pwd -P)/$(basename "$SFT_DATA")"
  local dataset_dir
  dataset_dir="$(mktemp -d "${TMPDIR:-/tmp}/llamafactory_sft_XXXXXX")"
  trap 'rm -rf -- "$dataset_dir"' EXIT
  mkdir -p "$RUN_DIR"

  cat > "${dataset_dir}/dataset_info.json" <<EOF
{
  "sft": {
    "file_name": "${SFT_DATA}",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF

  llamafactory-cli train \
    --stage sft --do_train true --model_name_or_path "$BASE_MODEL_DIR" \
    --dataset sft --dataset_dir "$dataset_dir" --template qwen3 \
    --finetuning_type lora --lora_target all --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --cutoff_len "$CUTOFF_LEN" --learning_rate "$SFT_LEARNING_RATE" \
    --num_train_epochs "$SFT_EPOCHS" --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing true --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" --val_size "$SFT_VAL_SIZE" \
    --eval_strategy steps --eval_steps "$EVAL_STEPS" --bf16 true --flash_attn auto \
    --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 5 \
    --save_strategy steps --save_steps 500 --save_total_limit 2 --plot_loss true \
    --output_dir "$SFT_LORA_DIR" --report_to none

  rm -rf -- "$dataset_dir"
  trap - EXIT
  echo "SFT LoRA training completed: ${SFT_LORA_DIR}"
}

main "$@"
