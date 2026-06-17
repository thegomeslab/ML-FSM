"""Golden-output regression tests for the Freezing String Method.

For each interpolation style, run a small deterministic FSM calculation with the
ASE EMT calculator and compare the resulting ``fsm.out`` against a stored golden
file (token- and tolerance-aware, see :mod:`tests.golden_utils`).  This pins the
numerical output so that an efficiency/flag change can be distinguished from a
change that actually alters energies or geometries.

Regenerating goldens
--------------------
When a change *intentionally* alters the output, regenerate and review the diff::

    pytest tests/test_golden.py --update-goldens
    git diff tests/data/golden

The golden files store the normalized text (timestamp masked), so they stay free
of wall-clock noise and diff cleanly in git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import GOLDEN, run_fsm
from tests.golden_utils import compare_lines, normalize_output

INTERPS = ["cart", "lst", "ric"]

# Small, fast, deterministic parameters (~a few seconds per style with EMT).
RUN_PARAMS: dict[str, Any] = {
    "optcoords": "cart",
    "method": "L-BFGS-B",
    "maxls": 3,
    "maxiter": 1,
    "nnodes_min": 5,
    "ninterp": 20,
    "calculator": "emt",
    "suffix": "golden",
}


def _run_and_read(reaction_dir: Path, interp: str) -> str:
    """Run the FSM for one interpolation style and return the normalized fsm.out."""
    run_fsm(reaction_dir, interp=interp, **RUN_PARAMS)
    outdir = next(reaction_dir.glob("fsm_interp_*_golden"))
    return normalize_output((outdir / "fsm.out").read_text())


@pytest.mark.parametrize("interp", INTERPS)
def test_golden_fsm_out(interp: str, reaction_dir: Path, update_goldens: bool) -> None:
    """fsm.out for each interpolation style must match its committed golden file."""
    actual = _run_and_read(reaction_dir, interp)
    golden_path = GOLDEN / f"diels_alder_{interp}" / "fsm.out"

    if update_goldens:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual)
        pytest.skip(f"regenerated golden for interp={interp}")

    assert golden_path.is_file(), f"missing golden {golden_path}; run with --update-goldens"
    diffs = compare_lines(golden_path.read_text(), actual)
    assert not diffs, "fsm.out diverged from golden:\n" + "\n".join(diffs[:10])
