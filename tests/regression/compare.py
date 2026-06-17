#!/usr/bin/env python
"""Compare two FSM fingerprints (see fingerprint.py) and report differences.

Distinguishes the two cases the regression check exists to tell apart:

* **Output change** — final coordinates or energies differ (beyond a tight
  tolerance), or the string structure (atoms/node count) changed.  This fails
  the check.
* **Efficiency-only change** — structures and energies match but the gradient
  count differs.  Reported as a note; it does *not* fail the check.

Because the baseline and candidate are produced on the same machine, the
comparison can be tight (no cross-platform float drift to absorb).

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


def _compare_one(interp: str, base: dict, cand: dict) -> tuple[list[str], list[str]]:
    """Return (failures, notes) for one interpolation style."""
    failures: list[str] = []
    notes: list[str] = []

    if base["symbols"] != cand["symbols"]:
        failures.append(f"[{interp}] atom symbols changed")
        return failures, notes

    n_base, n_cand = len(base["coords"]), len(cand["coords"])
    if n_base != n_cand:
        failures.append(f"[{interp}] node count changed: base={n_base} candidate={n_cand} (output change)")
        return failures, notes

    coord_diff = _max_abs_diff(base["coords"], cand["coords"])
    energy_diff = _max_abs_diff(base["energies"], cand["energies"])

    if coord_diff > COORD_ATOL:
        failures.append(f"[{interp}] coordinates changed: max |Δ| = {coord_diff:.3e} Å (tol {COORD_ATOL:.0e})")
    if energy_diff > ENERGY_ATOL:
        failures.append(f"[{interp}] energies changed: max |Δ| = {energy_diff:.3e} eV (tol {ENERGY_ATOL:.0e})")

    if base["ngrad"] != cand["ngrad"]:
        notes.append(
            f"[{interp}] gradient count changed: base={base['ngrad']} candidate={cand['ngrad']} (efficiency-only)"
        )

    return failures, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Baseline fingerprint JSON (e.g. from main).")
    parser.add_argument("candidate", type=Path, help="Candidate fingerprint JSON (e.g. from the PR).")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())["results"]
    cand = json.loads(args.candidate.read_text())["results"]

    all_failures: list[str] = []
    all_notes: list[str] = []
    for interp in INTERPS:
        if interp not in base or interp not in cand:
            all_failures.append(f"[{interp}] missing from one fingerprint")
            continue
        failures, notes = _compare_one(interp, base[interp], cand[interp])
        all_failures += failures
        all_notes += notes

    for note in all_notes:
        print(f"NOTE  {note}")

    if all_failures:
        print("\nFSM OUTPUT CHANGED relative to baseline:")
        for f in all_failures:
            print(f"  FAIL  {f}")
        print("\nIf this change is intended, this is expected — review the differences above.")
        return 1

    print("\nFSM output matches baseline (structures and energies unchanged).")
    if all_notes:
        print("Only efficiency (gradient count) differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
