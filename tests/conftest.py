"""Shared pytest fixtures and configuration for the ML-FSM test suite."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# Make run_fsm importable in-process. examples/ is an importable package, so add
# the repo root to sys.path and import it by its package-qualified name (this
# keeps a single module mapping for mypy: examples.fsm_example only).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fsm_example import run_fsm  # noqa: E402  (re-exported for tests)

__all__ = ["run_fsm"]

# Reuse the existing Diels-Alder example as test input rather than keeping a
# second copy in the repo. EMT supports its elements (C6H10) and it runs fast.
EXAMPLE_REACTION = REPO_ROOT / "examples" / "data" / "06_diels_alder"
REACTION_INPUTS = ("initial.xyz", "chg", "mult")


@pytest.fixture
def reaction_dir(tmp_path: Path) -> Path:
    """Copy the Diels-Alder reaction inputs into an isolated tmp directory.

    run_fsm writes its output subdirectory into the reaction directory, so the
    copy keeps the example tree clean across test runs.
    """
    dst = tmp_path / "diels_alder"
    dst.mkdir()
    for name in REACTION_INPUTS:
        shutil.copy(EXAMPLE_REACTION / name, dst / name)
    return dst
