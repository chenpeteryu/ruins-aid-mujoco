#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Codex currently exposes a read-only .git placeholder in the original workspace.
# This helper uses the project-local metadata directory. Normal clones use git.
exec git --git-dir="$PWD/.sync-repo" --work-tree="$PWD" "$@"
