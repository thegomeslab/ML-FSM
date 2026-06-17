"""Run the FSM in-process with an explicit Cartesian step size."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import run_fsm


def test_fsm_explicit_stepsize(reaction_dir: Path) -> None:
    """run_fsm completes when the step size is set explicitly (overriding nnodes_min)."""
    run_fsm(reaction_dir, calculator="emt", stepsize=0.2, ninterp=20, suffix="stepsize")
    outdir = next(reaction_dir.glob("fsm_interp_*_stepsize"))
    assert (outdir / "fsm.out").is_file()
    assert "Total gradient calls" in (outdir / "fsm.out").read_text()
