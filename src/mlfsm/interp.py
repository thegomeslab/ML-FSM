"""Interpolation methods for constructing paths between endpoint geometries."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.units import Bohr
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from mlfsm.coords import Redundant

angs_to_bohr = 1 / Bohr
deg_to_rad = np.pi / 180.0


@dataclass
class Interpolate:
    """Abstract base class for interpolation schemes between molecular geometries.

    Parameters
    ----------
    atoms1 : ase.Atoms
        Starting (reactant-side) geometry.
    atoms2 : ase.Atoms
        Ending (product-side) geometry.
    ninterp : int
        Number of discrete frames to generate along the interpolated path.
    gtol : float, optional
        Gradient tolerance passed to the L-BFGS-B minimizer (used by
        :class:`LST`). Default is 1e-4.
    return_q : bool, optional
        If ``True``, return interpolated internal coordinate vectors instead
        of Cartesian positions. Only meaningful for :class:`RIC`. Default is
        ``False``.
    raise_on_backtransf_fail : bool, optional
        If ``True``, raise a RuntimeError if the RIC back transformation fails
        to converge during interpolation or node growth. Only meaningful for
        :class:`RIC`. Default is ``True``.
    """

    atoms1: Atoms
    atoms2: Atoms
    ninterp: int
    gtol: float = 1e-4
    return_q: bool = False
    raise_on_backtransf_fail: bool = True

    def interpolate(self) -> NDArray[np.float32]:
        """Abstract interpolation routine — must be overridden by subclasses."""
        raise NotImplementedError

    def __call__(self) -> NDArray[np.float32]:
        """Call the interpolation routine and return interpolated geometries."""
        return self.interpolate()


class Linear(Interpolate):
    """Linear interpolation of Cartesian coordinates.

    Generates a reaction path by linearly blending the Cartesian positions of
    two endpoint geometries: ``(1-f)*xyz1 + f*xyz2`` for *f* in [0, 1]. Fast
    but may produce unphysical geometries for large conformational changes.
    """

    def interpolate(self) -> NDArray[np.float32]:
        """Return linearly interpolated Cartesian path.

        Returns
        -------
        NDArray[np.float32]
            Array of shape ``(ninterp, natoms*3)`` with flattened Cartesian
            coordinates for each interpolated frame.
        """
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()

        def xab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * xyz1 + f * xyz2, dtype=np.float64)

        fs = np.linspace(0, 1, self.ninterp)
        # build a 2D float32 array of shape (ninterp, natoms*3)
        return np.array([xab(f).flatten() for f in fs], dtype=np.float32)


class LST(Interpolate):
    """Linear Synchronous Transit (LST) interpolation.

    At each interpolation point *f*, minimizes the deviation from linearly
    interpolated pairwise interatomic distances while remaining close to the
    Cartesian linear blend. Produces more chemically realistic geometries than
    plain Cartesian interpolation at moderate extra cost.

    References
    ----------
    Halgren, T. A.; Lipscomb, W. N. *Chem. Phys. Lett.* **1977**, *49*, 225-232.
    """

    def obj(
        self,
        x_c: NDArray[np.floating],
        f: float,
        rab: Callable[[float], NDArray[np.floating]],
        xab: Callable[[float], NDArray[np.floating]],
    ) -> float:
        """Objective function minimized at each interpolation point.

        Computes the weighted sum of squared deviations of pairwise distances
        from their linearly interpolated target values, plus a small Cartesian
        restraint to prevent rigid-body drift.

        Parameters
        ----------
        x_c : NDArray
            Flattened trial Cartesian coordinates.
        f : float
            Interpolation parameter in [0, 1].
        rab : callable
            Returns linearly interpolated pairwise distances at parameter *f*.
        xab : callable
            Returns linearly interpolated Cartesian coordinates at parameter *f*.

        Returns
        -------
        float
            Scalar objective value.
        """
        x_c = x_c.reshape(-1, 3)
        rab_c = pdist(x_c)
        rab_i = rab(f)
        x_i = xab(f).reshape(-1, 3)

        return float((((rab_i - rab_c) ** 2) / rab_i**4).sum() + 5e-2 * ((x_i - x_c) ** 2).sum())

    def interpolate(self) -> NDArray[np.float32]:
        """Return LST-interpolated path.

        Returns
        -------
        NDArray[np.float32]
            Array of shape ``(ninterp, natoms*3)`` with flattened Cartesian
            coordinates for each frame.
        """
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        pdist_1 = pdist(xyz1)
        pdist_2 = pdist(xyz2)

        def rab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * pdist_1 + f * pdist_2, dtype=np.float64)

        def xab(f: float) -> NDArray[np.floating]:
            return np.array((1 - f) * xyz1 + f * xyz2, dtype=np.float64)

        minimize_kwargs = {
            "method": "L-BFGS-B",
            "options": {
                "gtol": self.gtol,
            },
        }
        string = [xab(0).flatten()]
        string += [
            minimize(self.obj, x0=xab(f).flatten(), args=(f, rab, xab), **minimize_kwargs).x  # type: ignore [call-overload]
            for f in np.linspace(0, 1, self.ninterp)[1:-1]
        ]
        string += [xab(1).flatten()]

        return np.array(string, dtype=np.float32)


@dataclass
class RIC(Interpolate):
    """Redundant internal coordinate (RIC) interpolation.

    Linearly interpolates between the reactant and product geometries in the
    space of delocalized redundant internal coordinates (bonds, angles,
    torsions, out-of-plane bends). The interpolated frames are then
    back-transformed to Cartesian coordinates via iterative application of the
    Wilson B-matrix. This scheme avoids the unphysical bond compressions that
    can occur with Cartesian interpolation for large conformational changes.

    Attributes
    ----------
    coords : Redundant
        The redundant internal coordinate system built from both endpoints.
    """

    def __post_init__(self) -> None:
        """Build the shared redundant internal coordinate system."""
        self.coords = Redundant(
            self.atoms1, self.atoms2, verbose=False, raise_on_backtransf_fail=self.raise_on_backtransf_fail
        )

    def interpolate(self) -> NDArray[np.float32]:
        """Return RIC-interpolated path in Cartesian (or internal) coordinates.

        Torsion periodicity wrapping is applied automatically when the
        interpolation crosses the ±π boundary.

        Returns
        -------
        NDArray[np.float32]
            If ``return_q`` is ``False`` (default): array of shape
            ``(ninterp, natoms*3)`` with flattened Cartesian coordinates.
            If ``return_q`` is ``True``: array of shape ``(ninterp, n_ics)``
            with internal coordinate values.
        """
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        q1 = self.coords.q(xyz1)  # type ignore[no-untyped-call]
        q2 = self.coords.q(xyz2)  # type ignore[no-untyped-call]
        dq = q2 - q1
        for i, name in enumerate(self.coords.keys):
            if ("tors" in name) and dq[i] > np.pi:
                while q1[i] < np.pi:
                    q1[i] += 2 * np.pi
            elif ("tors" in name) and dq[i] < -np.pi:
                while q2[i] < np.pi:
                    q2[i] += 2 * np.pi

        def xab(f: float) -> NDArray[np.float64]:
            return (1 - f) * q1 + f * q2

        if self.return_q:
            string = []
            for f in np.linspace(0, 1, self.ninterp):
                string.append(xab(f))
            return np.array(string, dtype=np.float32)

        xyz = xyz1
        string = []
        for f in np.linspace(0, 1, self.ninterp):
            xyz = self.coords.x(xyz, xab(f))  # type ignore[no-untyped-call]
            string.append(xyz)

        return np.array(string, dtype=np.float32)
