"""Unit tests for mlfsm.utils input loading and numeric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.constraints import FixAtoms

from mlfsm.utils import float_check, load_xyz, load_xyz_fixed
from tests.conftest import REACTIONS

DIELS_ALDER = REACTIONS / "diels_alder"


def test_float_check_passthrough_float() -> None:
    assert float_check(3.5) == 3.5


@pytest.mark.parametrize("value", [[2.0], (2.0,), np.array([2.0])])
def test_float_check_single_element_containers(value: object) -> None:
    assert float_check(value) == 2.0  # type: ignore[arg-type]


# Note: a 0-D ndarray hits len() of an unsized object and raises TypeError, despite
# the docstring claiming it is supported.
@pytest.mark.parametrize("value", [[1.0, 2.0], "x", None, np.array(2.0)])
def test_float_check_rejects_bad_input(value: object) -> None:
    with pytest.raises(TypeError):
        float_check(value)  # type: ignore[arg-type]


def test_load_xyz_returns_two_aligned_structures() -> None:
    reactant, product = load_xyz(DIELS_ALDER)
    assert len(reactant) == len(product) == 16
    # project_trans_rot aligns the product onto the reactant, so their centroids coincide.
    r_centroid = reactant.get_positions().mean(axis=0)
    p_centroid = product.get_positions().mean(axis=0)
    assert np.allclose(r_centroid, p_centroid, atol=1e-6)


def test_load_xyz_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="not found"):
        load_xyz(tmp_path)


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
