"""Unit tests for the interpolation schemes in mlfsm.interp."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfsm.interp import RIC, Linear
from mlfsm.utils import load_xyz
from tests.conftest import EXAMPLE_REACTION


@pytest.fixture
def endpoints() -> tuple[Atoms, Atoms]:
    return load_xyz(EXAMPLE_REACTION)


def test_linear_interpolate_shape_and_endpoints() -> None:
    a1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    a2 = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])
    ninterp = 10
    path = Linear(a1, a2, ninterp=ninterp).interpolate()

    assert path.shape == (ninterp, 2 * 3)
    assert path.dtype == np.float32
    # First and last frames reproduce the endpoint geometries.
    assert np.allclose(path[0], a1.get_positions().flatten(), atol=1e-6)
    assert np.allclose(path[-1], a2.get_positions().flatten(), atol=1e-6)


def test_ric_interpolate_return_q_endpoints(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    ric = RIC(reactant, product, ninterp=8, return_q=True)
    qpath = ric.interpolate()

    n_ics = len(ric.coords.keys)
    assert qpath.shape == (8, n_ics)
    # Internal-coordinate path starts at q(reactant) and ends at q(product),
    # up to the torsion ±pi wrapping applied inside interpolate().
    q1 = ric.coords.q(reactant.get_positions())
    q2 = ric.coords.q(product.get_positions())
    non_tors = [i for i, name in enumerate(ric.coords.keys) if "tors" not in name]
    assert np.allclose(qpath[0][non_tors], q1[non_tors], atol=1e-5)
    assert np.allclose(qpath[-1][non_tors], q2[non_tors], atol=1e-5)


def test_ric_interpolate_cartesian_shape(endpoints: tuple[Atoms, Atoms]) -> None:
    reactant, product = endpoints
    path = RIC(reactant, product, ninterp=6).interpolate()
    assert path.shape == (6, len(reactant), 3)
    assert path.dtype == np.float32


def _hooh(phi_deg: float) -> Atoms:
    """Hydrogen peroxide (H-O-O-H) with the H-O-O-H dihedral set to ``phi_deg``.

    The two oxygens lie on the x-axis; the terminal H atom is rotated about that
    axis so the only internal coordinate that differs between two ``_hooh`` calls
    is the dihedral.
    """
    phi = np.radians(phi_deg)
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([1.46, 0.0, 0.0])
    h1 = o1 + np.array([-0.5, 0.8, 0.0])
    h2 = o2 + np.array([0.5, 0.8 * np.cos(phi), 0.8 * np.sin(phi)])
    return Atoms("HOOH", positions=[h1, o1, o2, h2])


def test_ric_interpolate_wraps_torsion_across_pi() -> None:
    # Endpoints straddle the ±pi branch cut (+170 deg -> -170 deg). The physical
    # change is +20 deg; without periodicity wrapping the linear q-path would take
    # the ~340 deg long way around instead of the short arc.
    ric = RIC(_hooh(170.0), _hooh(190.0), ninterp=8, return_q=True)
    qpath = ric.interpolate()

    tors = [i for i, name in enumerate(ric.coords.keys) if "tors" in name]
    assert tors, "expected H-O-O-H to define a torsion coordinate"
    for i in tors:
        total = abs(qpath[-1, i] - qpath[0, i])
        assert 0 < total < np.pi  # short arc, not the wrapped long way
