#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env COMMONSENSE_BASE_MODEL
require_env COMMONSENSE_DATA_PATH

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commonsense

GPU="${1:-5}"
TARGET_SCORE="${TARGET_SCORE:-0.789537}"
RUN_ID="${GOLORA_RUN_ID:-golora-single-$(date '+%Y%m%d-%H%M%S')}"
TMUX_SESSION="${GOLORA_TMUX_SESSION:-golora-single}"
VARIANT_SPEC="${VARIANT_SPEC:?VARIANT_SPEC is required}"
BASE_MODEL="${BASE_MODEL:-./llama-7b-hf}"
MODEL_NAME="${MODEL_NAME:-LLaMA-7B}"
ADAPTER_NAME="${ADAPTER_NAME:-DoRA}"
TARGET_MODULES="${TARGET_MODULES:-[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]}"
PROMPT_STYLE="${PROMPT_STYLE:-alpaca}"
TRAIN_ON_INPUTS="${TRAIN_ON_INPUTS:-true}"
OUT_ROOT="${OUT_ROOT:-./finetuned_result/llama7_r_dora_ablation_parallel}"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

DOCS_DIR="${DOCS_DIR:-$(cd "$ROOT_DIR/.." && pwd)/docs/codex/$TMUX_SESSION}"
STATE_PATH="${GOLORA_STATE_PATH:-$DOCS_DIR/golora-state.json}"
JOURNAL_PATH="${GOLORA_JOURNAL_PATH:-$DOCS_DIR/golora-journal.md}"
NEXT_STEP_PATH="${GOLORA_NEXT_STEP_PATH:-$DOCS_DIR/golora-next-step.md}"
mkdir -p "$DOCS_DIR"

FORMAL_RESULTS="$OUT_ROOT/formal_results.tsv"
printf "variant\tweights_dir\taverage\tstatus\n" > "$FORMAL_RESULTS"

IFS='|' read -r name theta_clamp detach_trig warmup_steps ortho_direction geo_direction <<< "$VARIANT_SPEC"

pick_weights_path() {
  local outdir="$1"
  local state_path="$outdir/trainer_state.json"
  local best_ckpt=""
  local latest_ckpt

  if [[ ! -f "$state_path" ]]; then
    latest_ckpt="$(find "$outdir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [[ -n "${latest_ckpt:-}" && -f "$latest_ckpt/trainer_state.json" ]]; then
      state_path="$latest_ckpt/trainer_state.json"
    fi
  fi

  if [[ -f "$state_path" ]]; then
    best_ckpt="$(python - "$state_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    state = json.load(f)

print(state.get("best_model_checkpoint", ""))
PY
)"
  fi

  if [[ -n "${best_ckpt:-}" && -f "$best_ckpt/adapter_config.json" ]]; then
    echo "$best_ckpt"
    return 0
  fi

  latest_ckpt="$(find "$outdir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -n "${latest_ckpt:-}" && -f "$latest_ckpt/adapter_config.json" ]]; then
    echo "$latest_ckpt"
    return 0
  fi

  if [[ -f "$outdir/adapter_config.json" ]]; then
    echo "$outdir"
    return 0
  fi

  return 1
}

extract_average() {
  local weights_dir="$1"
  python experiment/collect_eval_average.py "$weights_dir" | tee "$weights_dir/average_summary.txt" >/dev/stderr
  python experiment/collect_eval_average.py --average-only "$weights_dir"
}

run_eval() {
  local weights_dir="$1"
  echo "[EvalScript] MODEL_NAME=$MODEL_NAME ADAPTER_NAME=$ADAPTER_NAME BASE_MODEL=$BASE_MODEL PROMPT_STYLE=$PROMPT_STYLE bash ./llama_7B_Dora_eval.sh $weights_dir $GPU"
  MODEL_NAME="$MODEL_NAME" ADAPTER_NAME="$ADAPTER_NAME" BASE_MODEL="$BASE_MODEL" PROMPT_STYLE="$PROMPT_STYLE" bash ./llama_7B_Dora_eval.sh "$weights_dir" "$GPU"
}

is_above_target() {
  local avg="$1"
  python - "$avg" "$TARGET_SCORE" <<'PY'
import sys
avg = float(sys.argv[1])
target = float(sys.argv[2])
raise SystemExit(0 if avg > target else 1)
PY
}

run_train() {
  local outdir="$1"

  rm -rf "$outdir"
  mkdir -p "$outdir"

  if [[ -n "$theta_clamp" ]]; then
    export DORA_GEO_THETA_CLAMP="$theta_clamp"
  else
    unset DORA_GEO_THETA_CLAMP || true
  fi

  local theta_mode="${DORA_GEO_THETA_CLAMP_MODE:-hard}"
  local anchor_mode="${DORA_GEO_ANCHOR_MODE:-base}"
  local anchor_momentum="${DORA_GEO_ANCHOR_MOMENTUM:-}"

  echo "[Train] variant=$name outdir=$outdir model_name=$MODEL_NAME adapter_name=$ADAPTER_NAME base_model=$BASE_MODEL prompt_style=$PROMPT_STYLE train_on_inputs=$TRAIN_ON_INPUTS target_modules=$TARGET_MODULES clamp=${theta_clamp:-none} theta_mode=$theta_mode anchor_mode=$anchor_mode anchor_momentum=${anchor_momentum:-default} detach=$detach_trig warmup=$warmup_steps ortho=$ortho_direction geo=$geo_direction num_epochs=2 gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python finetune.py \
    --base_model "$BASE_MODEL" \
    --data_path commonsense_170k.json \
    --output_dir "$outdir" \
    --batch_size 16 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --warmup_steps "$warmup_steps" \
    --cutoff_len 256 \
    --val_set_size 120 \
    --eval_step 80 \
    --save_step 80 \
    --adapter_name "$ADAPTER_NAME" \
    --target_modules "$TARGET_MODULES" \
    --lora_r 32 \
    --lora_alpha 64 \
    --train_on_inputs "$TRAIN_ON_INPUTS" \
    --prompt_style "$PROMPT_STYLE" \
    --use_gradient_checkpointing \
    --orthogonal_direction="$ortho_direction" \
    --geodesic_direction="$geo_direction" \
    --geodesic_detach_trig="$detach_trig"
}

run_status="failed"
run_decision="run exited before producing a final decision"
run_finalized=0
best_variant="$name"
best_weights_dir=""
best_avg=""

python experiment/golora_handoff.py start-run \
  --state-path "$STATE_PATH" \
  --journal-path "$JOURNAL_PATH" \
  --next-step-path "$NEXT_STEP_PATH" \
  --run-id "$RUN_ID" \
  --tmux-session "$TMUX_SESSION" \
  --gpu "$GPU" \
  --target-score "$TARGET_SCORE" \
  --repo-root "$ROOT_DIR" \
  --command "bash ./run_golora_single_variant.sh $GPU"

finish_run_handoff() {
  python experiment/golora_handoff.py finish-run \
    --state-path "$STATE_PATH" \
    --journal-path "$JOURNAL_PATH" \
    --next-step-path "$NEXT_STEP_PATH" \
    --run-id "$RUN_ID" \
    --tmux-session "$TMUX_SESSION" \
    --gpu "$GPU" \
    --target-score "$TARGET_SCORE" \
    --repo-root "$ROOT_DIR" \
    --run-status "$run_status" \
    --best-variant "$best_variant" \
    --best-average "$best_avg" \
    --best-weights-dir "$best_weights_dir" \
    --decision "$run_decision"
}

on_exit() {
  local exit_code=$?
  if [[ "$run_finalized" -eq 0 ]]; then
    finish_run_handoff
  fi
  exit "$exit_code"
}

trap on_exit EXIT

python experiment/golora_handoff.py start-variant \
  --state-path "$STATE_PATH" \
  --journal-path "$JOURNAL_PATH" \
  --next-step-path "$NEXT_STEP_PATH" \
  --run-id "$RUN_ID" \
  --tmux-session "$TMUX_SESSION" \
  --gpu "$GPU" \
  --target-score "$TARGET_SCORE" \
  --repo-root "$ROOT_DIR" \
  --variant "$name"

outdir="$OUT_ROOT/$name"
log_file="$LOG_ROOT/${name}.log"

{
  echo "===== $(date '+%F %T') ====="
  echo "[Variant] $name"
  run_train "$outdir"
  weights_dir="$(pick_weights_path "$outdir")"
  echo "[Weights] $weights_dir"
  run_eval "$weights_dir"
  avg="$(extract_average "$weights_dir")"
  echo "[Average] $avg"
  if is_above_target "$avg"; then
    status="PASS"
    variant_decision="variant=$name beats target=$TARGET_SCORE and remains a valid optimization candidate"
  else
    status="FAIL"
    variant_decision="variant=$name is below target=$TARGET_SCORE and must be reverted before the next redesign round"
  fi
  echo "[Decision] $variant_decision"
  printf "%s\t%s\t%s\t%s\n" "$name" "$weights_dir" "$avg" "$status" >> "$FORMAL_RESULTS"

  python experiment/golora_handoff.py finish-variant \
    --state-path "$STATE_PATH" \
    --journal-path "$JOURNAL_PATH" \
    --next-step-path "$NEXT_STEP_PATH" \
    --run-id "$RUN_ID" \
    --tmux-session "$TMUX_SESSION" \
    --gpu "$GPU" \
    --target-score "$TARGET_SCORE" \
    --repo-root "$ROOT_DIR" \
    --variant "$name" \
    --average "$avg" \
    --weights-dir "$weights_dir" \
    --decision "$variant_decision" \
    --variant-status "$status"
} 2>&1 | tee "$log_file"

best_weights_dir="$weights_dir"
best_avg="$avg"
if [[ "$status" == "PASS" ]]; then
  run_status="completed"
  run_decision="keep current optimization and continue from this design"
else
  run_status="failed"
  run_decision="average below target; revert optimization changes, re-analyze design, and rerun until average exceeds target"
fi

finish_run_handoff
run_finalized=1
