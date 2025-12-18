"""Utility functions for reading input and checking numeric types."""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms
from numpy.typing import NDArray

from mlfsm.geom import project_trans_rot, project_trans_rot_fixed


def load_xyz(reaction_dir: Path | str) -> tuple[Atoms, Atoms]:
    """
    Load reactant and product geometries from a directory.

    Assumes a file named initial.xyz containing the reactant and product geometries.
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

def load_xyz_fixed(reaction_dir: Path | str, fixed: NDArray[int]) -> tuple[Atoms, Atoms]:
    """
    Load reactant and product geometries from a directory.

    Assumes a file named initial.xyz containing the reactant and product geometries.
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
    """
    Convert scalars, 0D arrays, or single element containers to a float.

    Leaves floats alone. Raises an error for anything else.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
        return float(x[0])

    raise TypeError(f"Cannot convert safely to float: {x} ({type(x)})")
