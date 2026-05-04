#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CURRENT_SESSION="${CURRENT_SESSION:-golora-auto}"
CURRENT_RESULTS="${CURRENT_RESULTS:-./finetuned_result/llama7_r_dora_ablation_autoruns/formal_results.tsv}"
POLL_SECONDS="${POLL_SECONDS:-5}"
GPU5_SPEC="${GPU5_SPEC:-full_golora_g2_trigfix_theta0p20_seed42|0.20|False|100|true|true}"
GPU7_SPEC="${GPU7_SPEC:-full_golora_g2_trigfix_theta0p10_seed42|0.10|False|100|true|true}"

echo "[Splitter] watching $CURRENT_RESULTS for first completed variant"

while true; do
  if [[ -f "$CURRENT_RESULTS" ]]; then
    completed_count="$(tail -n +2 "$CURRENT_RESULTS" | wc -l | tr -d ' ')"
    if [[ "${completed_count:-0}" -ge 1 ]]; then
      break
    fi
  fi
  sleep "$POLL_SECONDS"
done

echo "[Splitter] first variant finished; stopping sequential session $CURRENT_SESSION"
tmux kill-session -t "$CURRENT_SESSION" 2>/dev/null || true

echo "[Splitter] launching remaining variants on GPU 5 and GPU 7"
SESSION_NAME="golora-theta0p20-gpu5" \
DOCS_DIR="$ROOT_DIR/../docs/codex/golora-theta0p20-gpu5" \
OUT_ROOT="./finetuned_result/llama7_r_dora_ablation_parallel/golora-theta0p20-gpu5" \
bash ./start_golora_single_variant_tmux.sh 5 "$GPU5_SPEC"

SESSION_NAME="golora-theta0p10-gpu7" \
DOCS_DIR="$ROOT_DIR/../docs/codex/golora-theta0p10-gpu7" \
OUT_ROOT="./finetuned_result/llama7_r_dora_ablation_parallel/golora-theta0p10-gpu7" \
bash ./start_golora_single_variant_tmux.sh 7 "$GPU7_SPEC"

echo "[Splitter] parallel launches started"
