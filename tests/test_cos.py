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
    assert np.isclose(string.stepsize, 0.5)
    assert string.nnodes_min == int(string.dist / 0.5)


def test_nnodes_min_matches_equivalent_stepsize(endpoints: tuple[Atoms, Atoms]) -> None:
    """nnodes_min=N and stepsize=D/N grow bitwise-identical RIC frontier nodes."""
    reactant, product = endpoints
    a = FreezingString(reactant, product, nnodes_min=5, interp_method="ric", ninterp=10)
    b = FreezingString(reactant, product, stepsize=a.stepsize, interp_method="ric", ninterp=10)
    a.grow()
    b.grow()
    for sa, sb in ((a.r_string, b.r_string), (a.p_string, b.p_string)):
        assert np.array_equal(sa[-1].get_positions(), sb[-1].get_positions())
    for ta, tb in ((a.r_tangent[-1], b.r_tangent[-1]), (a.p_tangent[-1], b.p_tangent[-1])):
        assert ta is not None
        assert tb is not None
        assert np.array_equal(ta, tb)
