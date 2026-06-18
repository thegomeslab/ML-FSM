"""Failure-mode tests for the RIC -> Cartesian back-transformation (coords.x).

The iterative B-matrix back-transformation is the fragile numerical kernel of the
RIC path: on a hard target it may not converge. ``MAX_ITERATIONS`` is monkeypatched
down so a normal Diels-Alder reactant->product step exhausts the iteration budget,
exercising both the raising and the fallback branches.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfsm import coords as coords_mod
from mlfsm.coords import Redundant
from mlfsm.utils import load_xyz
from tests.conftest import EXAMPLE_REACTION


@pytest.fixture
def endpoints() -> tuple[Atoms, Atoms]:
    return load_xyz(EXAMPLE_REACTION)


def test_backtransform_raises_on_nonconvergence(
    endpoints: tuple[Atoms, Atoms], monkeypatch: pytest.MonkeyPatch
) -> None:
    reactant, product = endpoints
    monkeypatch.setattr(coords_mod, "MAX_ITERATIONS", 1)
    coords = Redundant(reactant, product, raise_on_backtransf_fail=True)
    qtarget = coords.q(product.get_positions())
    with pytest.raises(RuntimeError, match="did not converge"):
        coords.x(reactant.get_positions(), qtarget)


def test_backtransform_returns_backup_when_not_raising(
    endpoints: tuple[Atoms, Atoms], monkeypatch: pytest.MonkeyPatch
) -> None:
    reactant, product = endpoints
    monkeypatch.setattr(coords_mod, "MAX_ITERATIONS", 1)
    coords = Redundant(reactant, product, raise_on_backtransf_fail=False)
    qtarget = coords.q(product.get_positions())
    xyz = coords.x(reactant.get_positions(), qtarget)
    # Instead of raising, x returns the best (backup) geometry it found.
    assert xyz.shape == reactant.get_positions().shape
    assert np.all(np.isfinite(xyz))
