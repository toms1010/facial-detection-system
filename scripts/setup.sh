#!/usr/bin/env bash
# One-shot setup: virtualenv, Python extras, native build.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Python environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[qt,dev]"

echo "==> Native hardware layer"
cmake -S . -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="$(command -v python)"
cmake --build cpp/build
ctest --test-dir cpp/build

echo
echo "Done. Start with:"
echo "  source .venv/bin/activate"
echo "  visionai self-test"
echo "  visionai"
