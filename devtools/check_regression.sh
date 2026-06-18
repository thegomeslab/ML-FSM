#!/usr/bin/env bash
#
# Check whether the current working tree changes FSM numerical output relative to
# a baseline ref (default: main).  Runs the same fingerprint script against the
# baseline's source and the current source, then compares structures/energies.
#
# Usage:
#   devtools/check_regression.sh [BASE_REF]
#   pixi run -e dev regression
#
# Requires the mlfsm runtime dependencies to be installed in the active
# environment (ase, numpy, scipy, geometric, networkx).  The package itself is
# imported directly from each checkout's src/ via PYTHONPATH, so no reinstall is
# needed.
set -euo pipefail

BASE_REF="${1:-main}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
BASE_WT="$TMP/base"

cleanup() {
  git -C "$ROOT" worktree remove --force "$BASE_WT" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

echo "Creating worktree for baseline ref '$BASE_REF'..."
git -C "$ROOT" worktree add -q --detach "$BASE_WT" "$BASE_REF"

echo "Fingerprinting baseline ($BASE_REF)..."
python "$ROOT/tests/regression/fingerprint.py" --src "$BASE_WT/src" --out "$TMP/base.json"

echo "Fingerprinting working tree..."
python "$ROOT/tests/regression/fingerprint.py" --src "$ROOT/src" --out "$TMP/pr.json"

echo "Comparing..."
python "$ROOT/tests/regression/compare.py" "$TMP/base.json" "$TMP/pr.json"
