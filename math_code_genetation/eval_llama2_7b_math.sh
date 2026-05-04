#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env MATHCODE_BASE_MODEL
require_env MATHCODE_GSM8K_TEST
require_env MATHCODE_MATH_TEST

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATH_DIR="$ROOT_DIR/math_code_genetation"
COMMONSENSE_DIR="$ROOT_DIR/commonsense_reasoning"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commonsense

GPU="${1:-6}"
BASE_MODEL_PATH="${MATHCODE_BASE_MODEL:-$ROOT_DIR/models/Llama-2-7b-hf}"
LORA_WEIGHTS_PATH="${MATHCODE_LORA_WEIGHTS:-$MATH_DIR/outputs/llama2_7b_golora_math}"
OUTPUT_DIR_PATH="${MATHCODE_RESULTS_DIR:-$MATH_DIR/results/llama2_7b_golora_math}"
GSM8K_PATH="${MATHCODE_GSM8K_TEST:-$MATH_DIR/data/gsm8k_test.jsonl}"
MATH_PATH="${MATHCODE_MATH_TEST:-$MATH_DIR/data/MATH_test.jsonl}"
mkdir -p "$OUTPUT_DIR_PATH"

CUDA_VISIBLE_DEVICES="$GPU" python "$MATH_DIR/evaluate_math.py" \
  --dataset gsm8k \
  --base_model "$BASE_MODEL_PATH" \
  --lora_weights "$LORA_WEIGHTS_PATH" \
  --dataset_path "$GSM8K_PATH" \
  --output_path "$OUTPUT_DIR_PATH/gsm8k_predictions.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" python "$MATH_DIR/evaluate_math.py" \
  --dataset MATH \
  --base_model "$BASE_MODEL_PATH" \
  --lora_weights "$LORA_WEIGHTS_PATH" \
  --dataset_path "$MATH_PATH" \
  --output_path "$OUTPUT_DIR_PATH/math_predictions.jsonl"
