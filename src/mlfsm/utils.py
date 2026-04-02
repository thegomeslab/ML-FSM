"""Utility functions for reading input and checking numeric types."""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from numpy.typing import NDArray

from mlfsm.geom import project_trans_rot, project_trans_rot_fixed


def load_xyz(reaction_dir: Path | str) -> tuple[Atoms, Atoms]:
    """Load reactant and product geometries from a reaction directory.

    Reads the first and last frames of ``initial.xyz`` and aligns them by
    minimizing rigid-body rotation and translation via the Kabsch algorithm.

    Parameters
    ----------
    reaction_dir : path-like
        Directory containing ``initial.xyz``.  The file must have at least two
        XYZ frames: the reactant (first) and the product (last).

    Returns
    -------
    reactant : ase.Atoms
        Reactant geometry after alignment.
    product : ase.Atoms
        Product geometry after alignment to the reactant.

    Raises
    ------
    Exception
        If ``initial.xyz`` does not exist in ``reaction_dir``.
    """
    xyz = Path(reaction_dir) / "initial.xyz"
    if not xyz.is_file():
        raise Exception(f"Input file {xyz} not found.")
    atoms = read(xyz, format="xyz", index=":")
    reactant = atoms[0]
    product = atoms[-1]
    r_xyz, p_xyz = project_trans_rot(reactant.get_positions(), product.get_positions())
    reactant.set_positions(r_xyz.reshape(-1, 3))
    product.set_positions(p_xyz.reshape(-1, 3))

    return reactant, product


def load_xyz_fixed(reaction_dir: Path | str, fixed: NDArray[np.integer[Any]]) -> tuple[Atoms, Atoms]:
    """Load reactant and product geometries with fixed-atom constraints applied.

    Reads ``initial.xyz``, aligns the structures, and attaches an ASE
    :class:`~ase.constraints.FixAtoms` constraint to both atoms objects so
    that the specified atoms are held stationary during FSM optimization.
    When ``fixed`` is empty the behaviour is identical to :func:`load_xyz`.

    Parameters
    ----------
    reaction_dir : path-like
        Directory containing ``initial.xyz``.
    fixed : NDArray[int]
        Zero-indexed array of atom indices to freeze.  Pass an empty array
        (``np.array([], dtype=int)``) to apply no constraints.

    Returns
    -------
    reactant : ase.Atoms
        Reactant geometry with :class:`~ase.constraints.FixAtoms` applied.
    product : ase.Atoms
        Product geometry with :class:`~ase.constraints.FixAtoms` applied.

    Raises
    ------
    Exception
        If ``initial.xyz`` does not exist in ``reaction_dir``.
    """
    xyz = Path(reaction_dir) / "initial.xyz"
    if not xyz.is_file():
        raise Exception(f"Input file {xyz} not found.")
    atoms = read(xyz, format="xyz", index=":")
    reactant = atoms[0]
    product = atoms[-1]
    if len(fixed) == 0:
        r_xyz, p_xyz = project_trans_rot(reactant.get_positions(), product.get_positions())
        reactant.set_positions(r_xyz.reshape(-1, 3))
        product.set_positions(p_xyz.reshape(-1, 3))
    else:
        c = FixAtoms(indices=fixed)
        r_xyz, p_xyz = project_trans_rot_fixed(reactant.get_positions(), product.get_positions(), fixed)
        reactant.set_positions(r_xyz.reshape(-1, 3))
        reactant.set_constraint(c)
        product.set_positions(p_xyz.reshape(-1, 3))
        product.set_constraint(c)

    return reactant, product


def float_check(x: float) -> float:
    """Convert a scalar, 0-D array, or single-element container to a Python float.

    Parameters
    ----------
    x : float, ndarray, list, or tuple
        Value to convert.  Must be a plain float, a 0-D NumPy array, or a
        length-1 sequence.

    Returns
    -------
    float
        The converted value.

    Raises
    ------
    TypeError
        If ``x`` is a multi-element container or an unsupported type.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
        return float(x[0])

    raise TypeError(f"Cannot convert safely to float: {x} ({type(x)})")
