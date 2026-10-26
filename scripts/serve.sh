#!/usr/bin/env bash
# One worker per GPU slice (ports 8001+) behind a load-balancing gateway on :8000 (which also serves the UI).
#   GPUS="MIG-... MIG-..." scripts/serve.sh      # restrict/choose slices (default: every GPU/MIG slice nvidia-smi lists)
#   HOST=0.0.0.0 PORT=8000 scripts/serve.sh
source "$(dirname "$0")/env.sh"; cd "$ROOT"; mkdir -p runs
unset CUDA_VISIBLE_DEVICES
if [ -z "${GPUS:-}" ]; then GPUS=$(nvidia-smi -L | grep -oE '(MIG|GPU)-[0-9a-f-]{36}' | awk '/^MIG/ {m=m" "$0; next} {g=g" "$0} END {print (m ? m : g)}'); fi
[ -z "$GPUS" ] && { echo "no GPUs found"; exit 1; }
pids=(); urls=(); i=0
for g in $GPUS; do
  port=$((8001 + i)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$g GPU_UUID=$g uvicorn refusal_probe.server.app:app --host 127.0.0.1 --port $port > "runs/worker-$port.log" 2>&1 &
  pids+=($!); urls+=("http://127.0.0.1:$port"); echo "worker :$port -> $g"
done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT INT TERM
WORKERS=$(IFS=,; echo "${urls[*]}") uvicorn refusal_probe.server.gateway:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
