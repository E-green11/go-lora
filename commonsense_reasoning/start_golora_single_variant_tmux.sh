#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${1:-5}"
VARIANT_SPEC="${2:?variant spec is required}"
IFS='|' read -r VARIANT_NAME _ <<< "$VARIANT_SPEC"
SAFE_VARIANT_NAME="${VARIANT_NAME//[^a-zA-Z0-9_-]/-}"
SESSION_NAME="${SESSION_NAME:-golora-${SAFE_VARIANT_NAME}-gpu${GPU}}"
TARGET_SCORE="${TARGET_SCORE:-0.789537}"
RUN_ID="${RUN_ID:-$SESSION_NAME-$(date '+%Y%m%d-%H%M%S')}"
DOCS_DIR="${DOCS_DIR:-$ROOT_DIR/../docs/codex/$SESSION_NAME}"
OUT_ROOT="${OUT_ROOT:-./finetuned_result/llama7_r_dora_ablation_parallel/$SESSION_NAME}"
THETA_CLAMP_MODE="${DORA_GEO_THETA_CLAMP_MODE:-hard}"
THETA_SOFTBAND_START="${DORA_GEO_THETA_SOFTBAND_START:-}"
ANCHOR_MODE="${DORA_GEO_ANCHOR_MODE:-base}"
ANCHOR_MOMENTUM="${DORA_GEO_ANCHOR_MOMENTUM:-}"
BASE_MODEL_PATH="${BASE_MODEL:-./llama-7b-hf}"
MODEL_NAME_VALUE="${MODEL_NAME:-LLaMA-7B}"
ADAPTER_NAME_VALUE="${ADAPTER_NAME:-DoRA}"
TARGET_MODULES_VALUE="${TARGET_MODULES:-[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]}"
PROMPT_STYLE_VALUE="${PROMPT_STYLE:-alpaca}"
TRAIN_ON_INPUTS_VALUE="${TRAIN_ON_INPUTS:-true}"
STATE_PATH="$DOCS_DIR/golora-state.json"
JOURNAL_PATH="$DOCS_DIR/golora-journal.md"
NEXT_STEP_PATH="$DOCS_DIR/golora-next-step.md"
RUN_COMMAND="bash ./run_golora_single_variant.sh $GPU"

mkdir -p "$DOCS_DIR"

python experiment/golora_handoff.py start-run \
  --state-path "$STATE_PATH" \
  --journal-path "$JOURNAL_PATH" \
  --next-step-path "$NEXT_STEP_PATH" \
  --run-id "$RUN_ID" \
  --tmux-session "$SESSION_NAME" \
  --gpu "$GPU" \
  --target-score "$TARGET_SCORE" \
  --repo-root "$ROOT_DIR" \
  --command "$RUN_COMMAND"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && export VARIANT_SPEC='$VARIANT_SPEC' OUT_ROOT='$OUT_ROOT' DOCS_DIR='$DOCS_DIR' BASE_MODEL='$BASE_MODEL_PATH' MODEL_NAME='$MODEL_NAME_VALUE' ADAPTER_NAME='$ADAPTER_NAME_VALUE' TARGET_MODULES='$TARGET_MODULES_VALUE' PROMPT_STYLE='$PROMPT_STYLE_VALUE' TRAIN_ON_INPUTS='$TRAIN_ON_INPUTS_VALUE' DORA_GEO_THETA_CLAMP_MODE='$THETA_CLAMP_MODE' DORA_GEO_THETA_SOFTBAND_START='$THETA_SOFTBAND_START' DORA_GEO_ANCHOR_MODE='$ANCHOR_MODE' DORA_GEO_ANCHOR_MOMENTUM='$ANCHOR_MOMENTUM' GOLORA_RUN_ID='$RUN_ID' GOLORA_TMUX_SESSION='$SESSION_NAME' GOLORA_TARGET_SCORE='$TARGET_SCORE' GOLORA_STATE_PATH='$STATE_PATH' GOLORA_JOURNAL_PATH='$JOURNAL_PATH' GOLORA_NEXT_STEP_PATH='$NEXT_STEP_PATH' && $RUN_COMMAND"

echo "Started GO-LoRA single-variant run in tmux session: $SESSION_NAME"
echo "Variant: $VARIANT_NAME"
echo "Output root: $OUT_ROOT"
echo "Read handoff files first:"
echo "  $STATE_PATH"
echo "  $JOURNAL_PATH"
echo "  $NEXT_STEP_PATH"
echo "Attach with: tmux attach -t $SESSION_NAME"
