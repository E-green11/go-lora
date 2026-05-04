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
require_env MATHCODE_HUMANEVAL_TEST
require_env MATHCODE_MBPP_TEST

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATH_DIR="$ROOT_DIR/math_code_genetation"
COMMONSENSE_DIR="$ROOT_DIR/commonsense_reasoning"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commonsense

GPU="${1:-7}"
BASE_MODEL_PATH="${MATHCODE_BASE_MODEL:-$ROOT_DIR/models/Llama-2-7b-hf}"
LORA_WEIGHTS_PATH="${MATHCODE_LORA_WEIGHTS:-$MATH_DIR/outputs/llama2_7b_golora_code}"
OUTPUT_DIR_PATH="${MATHCODE_RESULTS_DIR:-$MATH_DIR/results/llama2_7b_golora_code}"
HUMANEVAL_PATH="${MATHCODE_HUMANEVAL_TEST:-$MATH_DIR/data/HumanEval.jsonl}"
MBPP_PATH="${MATHCODE_MBPP_TEST:-$MATH_DIR/data/mbpp.test.jsonl}"
mkdir -p "$OUTPUT_DIR_PATH"

CUDA_VISIBLE_DEVICES="$GPU" python "$MATH_DIR/evaluate_code.py" \
  --dataset humaneval \
  --base_model "$BASE_MODEL_PATH" \
  --lora_weights "$LORA_WEIGHTS_PATH" \
  --dataset_path "$HUMANEVAL_PATH" \
  --output_path "$OUTPUT_DIR_PATH/humaneval_samples.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" python "$MATH_DIR/evaluate_code.py" \
  --dataset mbpp \
  --base_model "$BASE_MODEL_PATH" \
  --lora_weights "$LORA_WEIGHTS_PATH" \
  --dataset_path "$MBPP_PATH" \
  --output_path "$OUTPUT_DIR_PATH/mbpp_predictions.jsonl"
