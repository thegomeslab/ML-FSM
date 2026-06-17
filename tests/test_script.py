"""Smoke test: run the FSM end-to-end in-process with default RIC interpolation."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import run_fsm


def test_fsm_default_ric(reaction_dir: Path) -> None:
    """run_fsm completes on Diels-Alder with default parameters and EMT, writing fsm.out."""
    run_fsm(reaction_dir, calculator="emt", interp="ric", nnodes_min=5, ninterp=20, suffix="script")
    outdir = next(reaction_dir.glob("fsm_interp_*_script"))
    assert (outdir / "fsm.out").is_file()
    assert "Total gradient calls" in (outdir / "fsm.out").read_text()
