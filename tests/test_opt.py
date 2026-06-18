"""Tests for the node optimizers in mlfsm.opt."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from mlfsm.coords import Redundant
from mlfsm.opt import InternalsOptimizer
from mlfsm.utils import load_xyz
from tests.conftest import EXAMPLE_REACTION


@pytest.fixture
def endpoints() -> tuple[Atoms, Atoms]:
    return load_xyz(EXAMPLE_REACTION)


@pytest.mark.xfail(reason="InternalsOptimizer is under active development (see opt.py)", strict=False)
def test_internals_optimizer_relaxes_node(endpoints: tuple[Atoms, Atoms]) -> None:
    """A single internal-coordinate relaxation step returns a finite energy and geometry.

    Marked xfail because the internal-coordinate optimizer is not yet considered
    working; this records its status and will start passing (XPASS) once it is.
    """
    reactant, product = endpoints
    calc = EMT()
    coords = Redundant(reactant, product)
    optimizer = InternalsOptimizer(calc, "L-BFGS-B", maxiter=1, maxls=3, dmax=0.05)
    optimizer.coordsobj = coords

    tangent = (product.get_positions() - reactant.get_positions()).flatten()
    tangent /= np.linalg.norm(tangent)

    atomsf, energy, _nfev, _nit = optimizer.optimize(reactant, tangent)
    assert np.isfinite(energy)
    assert atomsf.get_positions().shape == reactant.get_positions().shape
