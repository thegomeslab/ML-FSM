"""Shared pytest fixtures and configuration for the ML-FSM test suite."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

# Make run_fsm importable in-process: examples/ is a script directory (no
# __init__.py), so add it to sys.path and import the top-level module. This
# matches how mypy resolves examples/fsm_example.py and avoids a duplicate
# module mapping (examples.fsm_example vs fsm_example).
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fsm_example import run_fsm  # noqa: E402  (re-exported for tests)

__all__ = ["run_fsm"]

TESTS_DATA = Path(__file__).parent / "data"
GOLDEN = TESTS_DATA / "golden"
REACTIONS = TESTS_DATA / "reactions"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --update-goldens flag for regenerating golden output files."""
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden output files instead of comparing against them.",
    )


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """True when golden files should be regenerated (--update-goldens or UPDATE_GOLDENS=1)."""
    return bool(request.config.getoption("--update-goldens")) or os.environ.get("UPDATE_GOLDENS") == "1"


@pytest.fixture
def reaction_dir(tmp_path: Path) -> Path:
    """Copy the frozen Diels-Alder reaction inputs into an isolated tmp directory.

    run_fsm writes its output subdirectory into the reaction directory, so the
    copy keeps the repository tree clean across test runs.
    """
    dst = tmp_path / "diels_alder"
    shutil.copytree(REACTIONS / "diels_alder", dst)
    return dst
