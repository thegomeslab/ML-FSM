"""Unit tests for FreezingString construction and interpolation dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfsm.coords import Redundant
from mlfsm.cos import FreezingString
from mlfsm.interp import LST, RIC, Linear
from mlfsm.utils import load_xyz
from tests.conftest import REACTIONS


@pytest.fixture
def endpoints() -> tuple[Atoms, Atoms]:
    return load_xyz(REACTIONS / "diels_alder")


@pytest.mark.parametrize(
    ("interp_method", "expected"),
    [("cart", Linear), ("lst", LST), ("ric", RIC)],
)
def test_interp_method_dispatch(endpoints: tuple[Atoms, Atoms], interp_method: str, expected: type) -> None:
    reactant, product = endpoints
    string = FreezingString(reactant, product, nnodes_min=5, interp_method=interp_method, ninterp=10)
    assert string.interp is expected


def test_bad_interp_method_raises(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    with pytest.raises(ValueError, match="Check interpolation method"):
        FreezingString(reactant, product, interp_method="bogus")


def test_ric_builds_init_coordsobj(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    string = FreezingString(reactant, product, nnodes_min=5, interp_method="ric", ninterp=10)
    assert isinstance(string.init_coordsobj, Redundant)


def test_non_ric_has_no_coordsobj(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    string = FreezingString(reactant, product, nnodes_min=5, interp_method="cart", ninterp=10)
    assert string.init_coordsobj is None


def test_explicit_stepsize_sets_cartesian_distance(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    string = FreezingString(reactant, product, interp_method="cart", ninterp=10, stepsize=0.5)
    assert string.use_cartesian_distance is True
    assert np.isclose(string.stepsize, 0.5)
    # nnodes_min is derived from the total Cartesian distance and the step size.
    assert string.nnodes_min == int(string.dist / 0.5)
