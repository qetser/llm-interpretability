#!/usr/bin/env bash
# Probe + benchmark every model in models.yaml (or the ones passed as args).
# Extra flags go to the pipeline, e.g.:  scripts/run_all.sh -- --lm-eval-limit 200 --judge
set -uo pipefail
source "$(dirname "$0")/env.sh"; cd "$ROOT"
models=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do models+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
[ ${#models[@]} -eq 0 ] && mapfile -t models < <(python -c "from refusal_probe.config import list_models; print('\n'.join(list_models()))")
for m in "${models[@]}"; do
  echo "=== $m ==="; mkdir -p runs
  python -m refusal_probe.pipeline --model "$m" "$@" 2>&1 | grep -vE "it/s\]|s/it\]" | tee "runs/$(basename "$m").log" || echo "FAILED: $m"
done
