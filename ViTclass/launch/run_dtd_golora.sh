#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU="${1:-1}"

if [[ -z "${VITCLASS_DATA_ROOT:-}" ]]; then
  echo "Missing required environment variable: VITCLASS_DATA_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHONPATH=. python ViTclass/train.py --config ViTclass/configs/dtd_golora.yaml
