#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATH_DIR="$ROOT_DIR/math_code_genetation"
COMMONSENSE_DIR="$ROOT_DIR/commonsense_reasoning"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commonsense

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env MATHCODE_BASE_MODEL
require_env MATHCODE_TRAIN_DATA

python "$MATH_DIR/prepare_data.py"

GPU="${1:-6}"
BASE_MODEL_PATH="${MATHCODE_BASE_MODEL}"
OUTPUT_DIR_PATH="${MATHCODE_OUT_DIR:-$MATH_DIR/outputs/llama2_7b_golora_math}"
TRAIN_DATA_PATH="${MATHCODE_TRAIN_DATA}"
TARGET_MODULES_REGEX="${MATHCODE_TARGET_MODULES:-model\\.layers\\.([0-9]|[1-2][0-9]|3[0-1])\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))}"
BATCH_SIZE_VALUE="${MATHCODE_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_VALUE="${MATHCODE_MICRO_BATCH_SIZE:-2}"
LORA_R_VALUE="${MATHCODE_LORA_R:-128}"
LORA_ALPHA_VALUE="${MATHCODE_LORA_ALPHA:-128}"
ORTHO_DIRECTION_VALUE="${MATHCODE_ORTHO_DIRECTION:-true}"
GEODESIC_DIRECTION_VALUE="${MATHCODE_GEODESIC_DIRECTION:-true}"
GEODESIC_DETACH_TRIG_VALUE="${MATHCODE_GEODESIC_DETACH_TRIG:-false}"
GEO_MEMORY_SAVE_MODE_VALUE="${MATHCODE_GEO_MEMORY_SAVE_MODE:-checkpoint_mlp_q_only}"

mkdir -p "$OUTPUT_DIR_PATH"

cd "$COMMONSENSE_DIR"
CUDA_VISIBLE_DEVICES="$GPU" DORA_GEO_MEMORY_SAVE_MODE="$GEO_MEMORY_SAVE_MODE_VALUE" python finetune.py \
  --base_model "$BASE_MODEL_PATH" \
  --data_path "$TRAIN_DATA_PATH" \
  --output_dir "$OUTPUT_DIR_PATH" \
  --batch_size "$BATCH_SIZE_VALUE" \
  --micro_batch_size "$MICRO_BATCH_SIZE_VALUE" \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_steps 0 \
  --weight_decay 0.0 \
  --cutoff_len 512 \
  --val_set_size 0 \
  --use_gradient_checkpointing \
  --adapter_name dora \
  --target_modules "$TARGET_MODULES_REGEX" \
  --lora_r "$LORA_R_VALUE" \
  --lora_alpha "$LORA_ALPHA_VALUE" \
  --lora_dropout 0.0 \
  --train_on_inputs true \
  --prompt_style alpaca \
  --group_by_length true \
  --orthogonal_direction="$ORTHO_DIRECTION_VALUE" \
  --geodesic_direction="$GEODESIC_DIRECTION_VALUE" \
  --geodesic_detach_trig="$GEODESIC_DETACH_TRIG_VALUE"
