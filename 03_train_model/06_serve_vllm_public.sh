#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM API server. Usage: bash 04_serve_vllm_public.sh /path/to/model

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR}"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6_27b_material}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"

[[ -n "${VLLM_API_KEY:-}" ]] || {
  echo "Set a strong API key first, for example:" >&2
  echo "export VLLM_API_KEY=\$(openssl rand -hex 32)" >&2
  exit 1
}
[[ -n "$MODEL_PATH" ]] || {
  echo "Usage: $0 /path/to/model" >&2
  exit 1
}
command -v vllm >/dev/null 2>&1 || {
  echo "Required command not found: vllm. Install it with: pip install -U vllm" >&2
  exit 1
}
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 1
}

args=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --api-key "$VLLM_API_KEY"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size 1
  --dtype auto
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --disable-log-requests
)
if [[ -n "$VLLM_QUANTIZATION" ]]; then
  args+=(--quantization "$VLLM_QUANTIZATION")
fi

echo "Starting OpenAI-compatible API at http://${HOST}:${PORT}/v1"
exec "${args[@]}"
