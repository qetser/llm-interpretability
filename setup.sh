#!/usr/bin/env bash
# Creates .venv (Python 3.11 via uv), a local Node under .node, and builds the web UI.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
[ -d .venv ] || uv venv --python 3.11 .venv
# torch comes from the cu124 index (driver here is CUDA 12.5); everything else from PyPI
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"
grep -v '^torch' requirements.txt | uv pip install --python .venv/bin/python -r /dev/stdin nodeenv
command -v node >/dev/null || { [ -d .node ] || .venv/bin/nodeenv --node=20.18.0 .node; export PATH="$PWD/.node/bin:$PATH"; }
(cd web && npm install && npm run build)
echo "done. Next: scripts/run_all.sh  then  scripts/serve.sh"
