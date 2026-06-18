"""Unit tests for mlfsm.utils input loading and numeric helpers."""

from __future__ import annotations

import numpy as np
from ase.constraints import FixAtoms

from mlfsm.utils import float_check, load_xyz, load_xyz_fixed
from tests.conftest import EXAMPLE_REACTION as DIELS_ALDER


def test_float_check_passthrough_float() -> None:
    assert float_check(3.5) == 3.5


def test_load_xyz_returns_two_aligned_structures() -> None:
    reactant, product = load_xyz(DIELS_ALDER)
    assert len(reactant) == len(product) == 16
    r_centroid = reactant.get_positions().mean(axis=0)
    p_centroid = product.get_positions().mean(axis=0)
    assert np.allclose(r_centroid, p_centroid, atol=1e-6)


def test_load_xyz_fixed_empty_matches_load_xyz() -> None:
    r_plain, _ = load_xyz(DIELS_ALDER)
    r_fixed, _ = load_xyz_fixed(DIELS_ALDER, fixed=np.array([], dtype=int))
    assert np.allclose(r_plain.get_positions(), r_fixed.get_positions())
    assert len(r_fixed.constraints) == 0


def test_load_xyz_fixed_attaches_constraint() -> None:
    fixed = np.array([0, 1, 2], dtype=int)
    reactant, product = load_xyz_fixed(DIELS_ALDER, fixed=fixed)
    for atoms in (reactant, product):
        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert set(atoms.constraints[0].get_indices()) == set(fixed)
