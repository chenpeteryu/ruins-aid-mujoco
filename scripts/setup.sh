#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
