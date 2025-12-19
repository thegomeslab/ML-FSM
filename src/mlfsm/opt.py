"""Optimization routines for FSM nodes.

Contains Cartesian and internal coordinate optimizers using projected gradients
and scipy-based minimization. Optimizers are used to refine node geometries
along a reaction path subject to constraints imposed by the FSM.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from numpy.typing import NDArray
from scipy.optimize import minimize

from mlfsm.coords import Redundant
from mlfsm.geom import generate_project_rt_tan


@dataclass
class Optimizer:
    """Base optimizer class for use with FSM node optimization.

    This abstract class provides an interface for optimizer implementations,
    requiring subclasses to define `obj` and `optimize` methods.
    """

    calc: Any
    method: str = "L-BFGS-B"
    maxiter: int = 3
    maxls: int = 2
    dmax: float = 0.3

    def obj(self, *args: Any, **kwargs: Any) -> Any:
        """Objective function to be implemented by subclasses.

        Args:
            xyz (ndarray): Cartesian coordinates (flattened).
            tangent (ndarray): Tangent vector used for projection.

        Raises
        ------
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError("No objective function")

    def optimizeobj(self, *args: Any, **kwargs: Any) -> Any:
        """Optimization method to be implemented by subclasses.

        Args:
            xyz (ndarray): Cartesian coordinates (flattened).
            tangent (ndarray): Tangent vector used for projection.

        Raises
        ------
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError("No optimize function")


class CartesianOptimizer(Optimizer):
    """Performs optimization in Cartesian coordinates using projected gradients."""

    def obj(self, xyz: NDArray[Any], tangent: NDArray[Any], atoms: Atoms) -> tuple[float, NDArray[Any]]:
        """Objective function for cartesian coordinate optimization.

        Computes energy and projected gradient given a position and tangent vector.

        Args:
            xyz (ndarray): Cartesian coordinates.
            tangent(ndarray): Tangent vector in Cartesian space.
            atoms (ase.Atoms): ASE atoms object with calculator.

        Returns
        -------
            tuple[float, ndarray]: Energy and projected gradient.
        """
        c = atoms.constraints
        if len(c) > 0:
            fixed_atoms = c[0].get_indices()
        else:
            fixed_atoms = np.array([], dtype=int)
        atoms.set_positions(xyz.reshape(-1, 3))
        atoms.calc = self.calc
        proj = generate_project_rt_tan(xyz.reshape(-1, 3), tangent)
        grads = -1 * atoms.get_forces()  # convert forces to grad
        for i in fixed_atoms:
            grads[i] = 0.0
        grads = grads.flatten()
        energy = atoms.get_potential_energy()
        pgrads = proj @ grads
        return energy, pgrads

    def optimize(self, atoms: Atoms, tangent: NDArray[Any]) -> tuple[Atoms, float, int]:
        """Run optimization in Cartesian coordinates using user specified method.

        Args:
            atoms (ASE.Atoms): ASE atoms object with calculator.
            tangent (ndarray): Tangent vector used for projection.

        Returns
        -------
            tuple[ASE.Atoms,float,int]: ASE.Atoms with final positions, energy of final structure, and number
            of gradient calculations used by optimization.
        """
        xyz = atoms.get_positions().flatten()
        config = {
            "fun": self.obj,
            "x0": xyz,
            "args": (tangent, atoms),
            "jac": True,
            "method": self.method,
            "bounds": [[j - self.dmax, j + self.dmax] for j in xyz],
            "options": {
                "maxiter": self.maxiter,
                "maxls": self.maxls,
            },
        }
        res = minimize(**config)
        atomsf = atoms.copy()
        atomsf.set_positions(res.x.reshape(-1, 3))
        return atomsf, res.fun, res.njev


@dataclass
class InternalsOptimizer(Optimizer):
    """Performs projected optimization in internal coordinates.

    NOTE: This optimizer is currently under development and may not work as expected.
    """

    maxls: int = 6
    dmax: float = 0.05

    def __post_init__(self) -> None:
        """Add coords, coordsobj, and angle_dmax."""
        self.coords = Redundant
        self.coordsobj: Any | None = None
        self.angle_dmax = self.dmax * 1.0

    def compute_bounds(self, q: NDArray[Any]) -> list[tuple[float, float]]:
        """Compute optimization bounds for each internal coordinate.

        Bounds are computed based on the coordinate type (e.g., bend, torsion),
        ensuring constraints appropriate to their periodicity and domain.

        Args:
            q (ndarray): Internal coordinate values.

        Returns
        -------
            list[tuple[float, float]]: List of bounds for each coordinate.
        """
        assert self.coordsobj is not None, "Coordsobj must be initialized"

        bounds = []
        for i, k in enumerate(self.coordsobj.keys):
            if "linearbnd" in k:
                angle_min = max(-np.pi, q[i] - self.angle_dmax)
                angle_max = min(np.pi, q[i] + self.angle_dmax)
                bounds += [(angle_min, angle_max)]
            elif "bend" in k:
                angle_min = max(0, q[i] - self.angle_dmax)
                angle_max = min(np.pi, q[i] + self.angle_dmax)
                bounds += [(angle_min, angle_max)]
            elif "tors" in k:
                angle_min = max(-np.pi, q[i] - self.angle_dmax)
                angle_max = min(np.pi, q[i] + self.angle_dmax)
                bounds += [(angle_min, angle_max)]
            elif "oop" in k:
                angle_min = max(-np.pi, q[i] - self.angle_dmax)
                angle_max = min(np.pi, q[i] + self.angle_dmax)
                bounds += [(angle_min, angle_max)]
            else:
                bounds += [(q[i] - self.dmax, q[i] + self.dmax)]

        return bounds

    def obj(
        self, q: NDArray[Any], xyzref: NDArray[Any], tangent: NDArray[Any], atoms: Atoms
    ) -> tuple[float, NDArray[Any]]:
        """Objective function for internal coordinate optimization.

        Computes the energy and projected gradient given internal
        coordinates and a tangent direction.

        Args:
            q (ndarray): Internal coordinates.
            xyzref (ndarray): Reference Cartesian coordinates.
            tangent (ndarray): Tangent vector in Cartesian space.
            atoms (ASE.Atoms): ASE atoms object with calculator.

        Returns
        -------
            tuple[float, ndarray]: Energy and projected gradient.
        """
        assert self.coordsobj is not None, "Coordsobj must be initialized"

        xyz = self.coordsobj.x(xyzref, q)
        atoms.set_positions(xyz.reshape(-1, 3))
        atoms.calc = self.calc
        proj = generate_project_rt_tan(xyz.reshape(-1, 3), tangent)
        grads = -1 * atoms.get_forces().flatten()  # convert forces to grad
        energy = atoms.get_potential_energy()
        pgrads = proj @ grads
        B = self.coordsobj.b_matrix(xyz)
        B_inv = np.linalg.pinv(B)
        BT_inv = np.linalg.pinv(B.T)
        P = B @ B_inv
        pgrads = P @ BT_inv @ pgrads

        return energy, pgrads

    def optimize(self, atoms: Atoms, tangent: NDArray[Any]) -> tuple[Atoms, float, int]:
        """Run optimization in internal coordinates using user specified method.

        Args:
            atoms (ASE.Atoms): ASE atoms object with calculator.
            tangent (ndarray): Tangent vector used for projection.

        Returns
        -------
            tuple[ASE.Atoms,float,int]: ASE.Atoms with final positions, energy of final structure, and number
            of gradient calculations used by optimization.
        """
        assert self.coordsobj is not None, "Coordsobj must be initialized"

        q = self.coordsobj.q(atoms.get_positions())
        xyz = atoms.get_positions()
        config = {
            "fun": self.obj,
            "x0": q,
            "args": (xyz, tangent, atoms),
            "jac": True,
            "method": self.method,
            "bounds": self.compute_bounds(q),
            "options": {"maxiter": self.maxiter, "maxls": self.maxls, "iprint": 0},
        }
        res = minimize(**config)
        xf = self.coordsobj.x(xyz, res.x)
        atomsf = atoms.copy()
        atomsf.set_positions(xf)

        return atomsf, res.fun, res.njev
