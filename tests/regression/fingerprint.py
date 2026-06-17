#!/usr/bin/env python
"""Produce a compact numerical fingerprint of FSM output for regression checks.

This script runs a small, deterministic Freezing String Method calculation with
the ASE EMT calculator for each interpolation style and records the final
optimized string (per-node coordinates and energies) plus the gradient count.

It is deliberately driver-stable: it imports only the core ``mlfsm`` API and can
be pointed at an arbitrary checkout's source tree via ``--src``.  The
compare-against-baseline workflow runs this *same* script against both the base
branch and the PR branch (see ``tests/regression/compare.py`` and
``.github/workflows/regression.yml``), so any difference in the recorded
structures/energies is attributable to the code change under review.

Usage
-----
    python tests/regression/fingerprint.py --src path/to/src --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTERPS = ["cart", "lst", "ric"]

# Small, fast, deterministic run parameters (a few seconds per style with EMT).
NNODES_MIN = 5
NINTERP = 20
MAXITER = 1
MAXLS = 3
DMAX = 0.05

HERE = Path(__file__).resolve().parent
DEFAULT_REACTION = HERE.parent / "data" / "reactions" / "diels_alder"


def _fingerprint_one(reaction_dir: Path, interp: str) -> dict:
    """Run the FSM for a single interpolation style and return its fingerprint."""
    from ase.calculators.emt import EMT

    from mlfsm.cos import FreezingString
    from mlfsm.opt import CartesianOptimizer
    from mlfsm.utils import load_xyz

    reactant, product = load_xyz(reaction_dir)
    calc = EMT()
    reactant.calc = calc
    product.calc = calc

    string = FreezingString(reactant, product, nnodes_min=NNODES_MIN, interp_method=interp, ninterp=NINTERP)
    optimizer = CartesianOptimizer(calc, "L-BFGS-B", MAXITER, MAXLS, DMAX)

    while string.growing:
        string.grow()
        string.optimize(optimizer)

    path = string.r_string + string.p_string[::-1]
    energies = string.r_energy + string.p_energy[::-1]
    return {
        "symbols": path[0].get_chemical_symbols(),
        "coords": [atoms.get_positions().tolist() for atoms in path],
        "energies": list(energies),
        "ngrad": int(string.ngrad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="mlfsm source tree to import (prepended to sys.path).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where to write the fingerprint JSON.")
    parser.add_argument("--reaction", type=Path, default=DEFAULT_REACTION, help="Reaction input directory.")
    args = parser.parse_args()

    if args.src is not None:
        sys.path.insert(0, str(args.src.resolve()))

    result = {interp: _fingerprint_one(args.reaction, interp) for interp in INTERPS}

    import mlfsm

    payload = {"mlfsm_version": getattr(mlfsm, "__version__", "unknown"), "results": result}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote fingerprint for mlfsm {payload['mlfsm_version']} -> {args.out}")


if __name__ == "__main__":
    main()
