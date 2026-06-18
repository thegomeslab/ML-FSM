"""End-to-end FSM runs driven by the Cartesian optimizer (EMT).

Covers both interpolants the user expects to keep working alongside RIC (cart and
LST) and both scipy minimizers used in practice: the bounded L-BFGS-B (default,
respects the per-node trust radius) and the gradient-only CG.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import run_fsm


@pytest.mark.parametrize(("interp", "method"), [("cart", "L-BFGS-B"), ("lst", "CG")])
def test_fsm_cartesian_optimizer(reaction_dir: Path, interp: str, method: str) -> None:
    """run_fsm completes with the Cartesian optimizer for cart/L-BFGS-B and lst/CG."""
    suffix = f"{interp}_{method}"
    run_fsm(
        reaction_dir,
        calculator="emt",
        interp=interp,
        method=method,
        nnodes_min=5,
        ninterp=20,
        suffix=suffix,
    )
    outdir = next(reaction_dir.glob(f"fsm_interp_*_{suffix}"))
    assert (outdir / "fsm.out").is_file()
    assert "Total gradient calls" in (outdir / "fsm.out").read_text()
