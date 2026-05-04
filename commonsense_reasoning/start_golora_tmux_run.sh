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

GPU="${1:-5}"
SESSION_NAME="${SESSION_NAME:-golora-auto}"
TARGET_SCORE="${TARGET_SCORE:-0.789537}"
RUN_ID="${RUN_ID:-golora-auto-$(date '+%Y%m%d-%H%M%S')}"
DOCS_DIR="${DOCS_DIR:-$ROOT_DIR/../docs/codex}"
STATE_PATH="$DOCS_DIR/golora-state.json"
JOURNAL_PATH="$DOCS_DIR/golora-journal.md"
NEXT_STEP_PATH="$DOCS_DIR/golora-next-step.md"
RUN_COMMAND="bash ./run_golora_optimization_rounds.sh $GPU"

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
  "cd '$ROOT_DIR' && export GOLORA_RUN_ID='$RUN_ID' GOLORA_TMUX_SESSION='$SESSION_NAME' GOLORA_TARGET_SCORE='$TARGET_SCORE' GOLORA_STATE_PATH='$STATE_PATH' GOLORA_JOURNAL_PATH='$JOURNAL_PATH' GOLORA_NEXT_STEP_PATH='$NEXT_STEP_PATH' && $RUN_COMMAND"

echo "Started GO-LoRA autorun in tmux session: $SESSION_NAME"
echo "Read handoff files first:"
echo "  $STATE_PATH"
echo "  $JOURNAL_PATH"
echo "  $NEXT_STEP_PATH"
echo "Attach with: tmux attach -t $SESSION_NAME"
