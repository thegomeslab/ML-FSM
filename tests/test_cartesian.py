"""Run the FSM in-process with LST interpolation and the CG optimizer."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import run_fsm


def test_fsm_lst_cg(reaction_dir: Path) -> None:
    """run_fsm completes with LST interpolation and the CG optimizer using EMT."""
    run_fsm(
        reaction_dir,
        calculator="emt",
        interp="lst",
        method="CG",
        nnodes_min=5,
        ninterp=20,
        suffix="cartesian",
    )
    outdir = next(reaction_dir.glob("fsm_interp_*_cartesian"))
    assert (outdir / "fsm.out").is_file()
    assert "Total gradient calls" in (outdir / "fsm.out").read_text()
