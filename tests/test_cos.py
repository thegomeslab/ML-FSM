"""Unit tests for FreezingString construction and interpolation dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfsm.cos import FreezingString
from mlfsm.utils import load_xyz
from tests.conftest import EXAMPLE_REACTION


@pytest.fixture
def endpoints() -> tuple[Atoms, Atoms]:
    return load_xyz(EXAMPLE_REACTION)


def test_bad_interp_method_raises(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    with pytest.raises(ValueError, match="Check interpolation method"):
        FreezingString(reactant, product, interp_method="bogus")


def test_explicit_stepsize_sets_cartesian_distance(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    string = FreezingString(reactant, product, interp_method="cart", ninterp=10, stepsize=0.5)
    assert string.use_cartesian_distance is True
    assert np.isclose(string.stepsize, 0.5)
    assert string.nnodes_min == int(string.dist / 0.5)
