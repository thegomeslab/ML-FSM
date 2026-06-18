#!/usr/bin/env python
"""Compare two FSM fingerprints (see fingerprint.py) and report output changes.

The FSM is deterministic, so for a given input the final string is fully
determined by the source code.  A change either leaves the output identical or
it changes the structures/energies — there is no "efficiency-only" middle
ground, since an identical path implies an identical amount of work.  This
script therefore reports a single thing: did the FSM output change?

Because the baseline and candidate are produced on the same machine, the
comparison is tight (no cross-platform float drift to absorb).

Usage
-----
    python tests/regression/compare.py base.json pr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same-runner determinism means repeated runs are effectively bit-identical;
# this tolerance only guards against trivial float non-associativity.
COORD_ATOL = 1e-8
ENERGY_ATOL = 1e-8

INTERPS = ["cart", "lst", "ric"]


def _max_abs_diff(a: list, b: list) -> float:
    """Maximum absolute elementwise difference between two nested numeric lists."""
    import numpy as np

    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _compare_one(interp: str, base: dict, cand: dict) -> list[str]:
    """Return the list of output changes for one interpolation style."""
    if base["symbols"] != cand["symbols"]:
        return [f"[{interp}] atom symbols changed"]

    n_base, n_cand = len(base["coords"]), len(cand["coords"])
    if n_base != n_cand:
        return [f"[{interp}] node count changed: base={n_base} candidate={n_cand}"]

    failures: list[str] = []
    coord_diff = _max_abs_diff(base["coords"], cand["coords"])
    energy_diff = _max_abs_diff(base["energies"], cand["energies"])
    if coord_diff > COORD_ATOL:
        failures.append(f"[{interp}] coordinates changed: max |Δ| = {coord_diff:.3e} Å (tol {COORD_ATOL:.0e})")
    if energy_diff > ENERGY_ATOL:
        failures.append(f"[{interp}] energies changed: max |Δ| = {energy_diff:.3e} eV (tol {ENERGY_ATOL:.0e})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Baseline fingerprint JSON (e.g. from main).")
    parser.add_argument("candidate", type=Path, help="Candidate fingerprint JSON (e.g. from the PR).")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())["results"]
    cand = json.loads(args.candidate.read_text())["results"]

    failures: list[str] = []
    for interp in INTERPS:
        if interp not in base or interp not in cand:
            failures.append(f"[{interp}] missing from one fingerprint")
            continue
        failures += _compare_one(interp, base[interp], cand[interp])

    if failures:
        print("FSM OUTPUT CHANGED relative to baseline:")
        for f in failures:
            print(f"  FAIL  {f}")
        print("\nIf this change is intended, this is expected — review the differences above.")
        return 1

    print("FSM output matches baseline (structures and energies unchanged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
