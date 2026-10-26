# source me. Pins the GPU: this box is MIG-partitioned, so a process only sees one slice.
# Override with MIG_DEVICE=<uuid|index> (see `nvidia-smi -L`).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/.venv/bin:$ROOT/.node/bin:$PATH"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${MIG_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
export TOKENIZERS_PARALLELISM=false
