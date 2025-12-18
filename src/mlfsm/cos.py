"""Freezing String Method driver for reaction pathway construction."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from ase import Atoms
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mlfsm.coords import Cartesian
from mlfsm.geom import (
    calculate_arc_length,
    distance,
    normalize,
    project_trans_rot,
    project_trans_rot_fixed,
)
from mlfsm.interp import LST, RIC, Linear
from mlfsm.utils import float_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreezingString:
    """Implements the Freezing String Method."""

    def __init__(
        self,
        reactant: Atoms,
        product: Atoms,
        nnodes_min: int = 10,
        interp_method: str = "ric",
        ninterp: int = 100,
        stepsize: float = 0.0,
    ) -> None:
        self.interp: Any
        self.interp_method = interp_method
        self.nnodes_min = int(nnodes_min)
        self.ninterp = int(ninterp)
        self.use_cartesian_distance = True if stepsize > 0 else False

        if interp_method == "cart":
            self.interp = Linear
        elif interp_method == "lst":
            self.interp = LST
        elif interp_method == "ric":
            self.interp = RIC
        else:
            raise ValueError("Check interpolation method")

        self.atoms = reactant.copy()
        self.natoms = len(self.atoms.numbers)

        if not self.use_cartesian_distance:
            interp = self.interp(reactant, product, ninterp=self.ninterp)
            s = calculate_arc_length(interp())
            self.dist = s[-1]
            self.stepsize = self.dist / self.nnodes_min
        else:
            interp = Linear(reactant, product, ninterp=self.ninterp)
            s = calculate_arc_length(interp())
            self.dist = s[-1]
            self.stepsize = float(stepsize)
            self.nnodes_min = int(self.dist / self.stepsize)

        logger.info(f"NNODES_MIN: {self.nnodes_min}")
        logger.info(f"DIST: {self.dist:.3f} STEPSIZE: {self.stepsize:.3f}")

        self.r_string: list[Atoms] = [reactant.copy()]
        self.r_fix: list[bool] = [True]
        self.r_energy: list[Optional[float]] = [None]
        self.r_tangent: list[Optional[NDArray[Any]]] = [None]
        self.r_nnodes = len(self.r_string)
        self.p_string: list[Atoms] = [product.copy()]
        self.p_fix: list[bool] = [True]
        self.p_energy: list[Optional[float]] = [None]
        self.p_tangent: list[Optional[NDArray[Any]]] = [None]
        self.p_nnodes = len(self.p_string)

        self.growing = True
        self.iteration = 0
        self.ngrad = 0

        self.coordsobj: Any = None

    def interpolate(self, outdir: Path | str) -> None:
        """Generate and write interpolated string between current endpoints."""
        outfile = Path(outdir) / f"interp_{self.iteration:02d}.xyz"

        r_atoms = self.r_string[-1]
        p_atoms = self.p_string[-1]

        r_xyz, p_xyz = project_trans_rot(r_atoms.get_positions(), p_atoms.get_positions())
        r_xyz, p_xyz = r_xyz.flatten(), p_xyz.flatten()

        interp = self.interp(r_atoms, p_atoms, ninterp=self.ninterp)
        string = interp()
        s = calculate_arc_length(string)

        path = []
        for i in range(self.ninterp):
            atoms = self.atoms.copy()
            atoms.set_positions(string[i].reshape(-1, 3))
            path.append(atoms)

        with outfile.open("w") as f:
            for i, atoms in enumerate(path):
                f.write(f"{self.natoms}\n")
                f.write(f"{s[i]:.5f}\n")
                for atom, xyz in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
                    x, y, z = map(float, xyz)
                    f.write(f"{atom} {x:.8f} {y:.8f} {z:.8f}\n")

    def grow(self) -> None:
        """Grow the string by adding nodes from each end."""
        r_atoms = self.r_string[-1]
        p_atoms = self.p_string[-1]

        r_xyz, p_xyz = project_trans_rot(r_atoms.get_positions(), p_atoms.get_positions())
        r_xyz, p_xyz = r_xyz.flatten(), p_xyz.flatten()

        return_q = self.use_cartesian_distance
        interp = self.interp(r_atoms, p_atoms, ninterp=self.ninterp, return_q=return_q)
        try:
            self.coordsobj = interp.coords
        except Exception:
            self.coordsobj = Cartesian(r_atoms, p_atoms)

        if self.use_cartesian_distance and self.interp_method == "ric":
            string = interp()
            s = calculate_arc_length(string)
            cs = CubicSpline(s, string, axis=0)

            self.dist = distance(r_xyz, p_xyz)
            if self.dist < self.stepsize:
                self.growing = False
                return

            r_prev = r_xyz.copy().reshape(-1, 3)
            r_idx = 1
            for qtarget in string[1:-1]:
                r_next = interp.coords.x(r_prev, qtarget)
                _, r_next = project_trans_rot(r_xyz.reshape(-1, 3), r_next)
                r_next = r_next.reshape(-1, 3)
                r_s = distance(r_xyz, r_next)
                if r_s > self.stepsize:
                    break
                r_prev = r_next.copy()
                r_idx += 1

            r_frontier = self.atoms.copy()
            r_frontier.set_positions(r_next.reshape(-1, 3))

            dqds = cs(s[r_idx], 1)
            Bprim = interp.coords.b_matrix(r_next)
            U = interp.coords.u_matrix(Bprim)
            B = U.T @ Bprim
            BT_inv = np.linalg.pinv(B @ B.T) @ B
            dqds = U.T @ dqds
            dxds = BT_inv.T @ dqds

            self.r_string += [r_frontier]
            self.r_fix += [False]
            self.r_energy += [None]
            self.r_tangent += [normalize(dxds)]
            self.r_nnodes = len(self.r_string)

            if self.dist <= 2 * self.stepsize:
                self.growing = False
                return

            p_prev = p_xyz.copy().reshape(-1, 3)
            p_idx = 1
            for qtarget in string[1:-1][::-1]:
                p_next = interp.coords.x(p_prev, qtarget)
                _, p_next = project_trans_rot(p_xyz.reshape(-1, 3), p_next)
                p_next = p_next.reshape(-1, 3)
                p_s = distance(p_xyz, p_next)
                if p_s > self.stepsize:
                    break
                p_prev = p_next.copy()
                p_idx += 1

            p_frontier = self.atoms.copy()
            p_frontier.set_positions(p_next.reshape(-1, 3))

            dqds = cs(s[p_idx], 1)
            Bprim = interp.coords.b_matrix(p_next)
            U = interp.coords.u_matrix(Bprim)
            B = U.T @ Bprim
            BT_inv = np.linalg.pinv(B @ B.T) @ B
            dqds = U.T @ dqds
            dxds = BT_inv.T @ dqds

            self.p_string += [p_frontier]
            self.p_fix += [False]
            self.p_energy += [None]
            self.p_tangent += [normalize(dxds)]
            self.p_nnodes = len(self.p_string)

        else:
            string = interp()
            s = calculate_arc_length(string)
            cs = CubicSpline(s, string.reshape(self.ninterp, 3 * self.natoms), axis=0)
            self.dist = s[-1]

            if self.dist < self.stepsize:
                self.growing = False
                return

            r_idx = np.abs(s - self.stepsize).argmin()
            p_idx = np.abs(s - (s[-1] - self.stepsize)).argmin()
            r_frontier = self.atoms.copy()
            r_frontier.set_positions(string[r_idx].reshape(-1, 3))

            self.r_string += [r_frontier]
            self.r_fix += [False]
            self.r_energy += [None]
            self.r_tangent += [normalize(cs(s[r_idx], 1))]
            self.r_nnodes = len(self.r_string)

            if self.dist <= 2 * self.stepsize:
                self.growing = False
                return

            p_frontier = self.atoms.copy()
            p_frontier.set_positions(string[p_idx].reshape(-1, 3))

            self.p_string += [p_frontier]
            self.p_fix += [False]
            self.p_energy += [None]
            self.p_tangent += [normalize(cs(s[p_idx], 1))]
            self.p_nnodes = len(self.p_string)

    def optimize(self, optimizer: Any) -> None:
        """Relax unfixed nodes on the hyperplane orthogonal to the local tangent direction."""
        self.iteration += 1
        optimizer.coordsobj = self.coordsobj

        for i in range(self.r_nnodes):
            if self.r_energy[i] is None and self.r_fix[i]:
                energy = optimizer.calc.get_potential_energy(self.r_string[i])
                self.r_energy[i] = float_check(energy)
            elif not self.r_fix[i]:
                assert self.r_tangent[i] is not None
                atoms = self.r_string[i]
                try:
                    atoms, energy, ngrad = optimizer.optimize(atoms, self.r_tangent[i])
                    self.r_string[i] = atoms
                    self.r_energy[i] = float_check(energy)
                except Exception:
                    energy = optimizer.calc.get_potential_energy(atoms)
                    self.r_energy[i] = float_check(energy)
                    ngrad = 0
                self.r_fix[i] = True
                self.ngrad += ngrad

        for i in range(self.p_nnodes):
            if self.p_energy[i] is None and self.p_fix[i]:
                energy = optimizer.calc.get_potential_energy(self.p_string[i])
                self.p_energy[i] = float_check(energy)
            elif not self.p_fix[i]:
                assert self.p_tangent[i] is not None
                atoms = self.p_string[i]
                try:
                    atoms, energy, ngrad = optimizer.optimize(atoms, self.p_tangent[i])
                    self.p_string[i] = atoms
                    self.p_energy[i] = float_check(energy)
                except Exception:
                    energy = optimizer.calc.get_potential_energy(atoms)
                    self.p_energy[i] = float_check(energy)
                    ngrad = 0
                self.p_fix[i] = True
                self.ngrad += ngrad

        self.dist = distance(self.r_string[-1].get_positions().flatten(), self.p_string[-1].get_positions().flatten())

        if self.dist < self.stepsize:
            self.growing = False

    def write(self, outdir: Path | str) -> None:
        """Write current string geometries and relative energies to an XYZ file."""
        outdir = Path(outdir)
        outfile = outdir / f"vfile_{self.iteration:02d}.xyz"
        gradfile = outdir / "ngrad.txt"

        path = self.r_string + self.p_string[::-1]
        string = np.stack([atoms.get_positions() for atoms in path], axis=0)
        s = calculate_arc_length(string)
        energy = np.array(self.r_energy + self.p_energy[::-1])
        energy = energy - energy.min()  # now will be in just eV

        # check for fixed atoms
        c = path[0].constraints
        if len(c) > 0:
            fixed = True
            fixed_atoms = c[0].get_indices()
        else:
            fixed = False

        with outfile.open("w") as f:
            for i, atoms in enumerate(path):
                if fixed:
                    _, xyz = project_trans_rot_fixed(string[0], string[i], fixed=fixed_atoms)
                else:
                    _, xyz = project_trans_rot(string[0], string[i])
                xyz = xyz.reshape(-1, 3)
                f.write(f"{self.natoms}\n")
                f.write(f"{s[i]:.5f} {energy[i]:.3f}\n")
                for atom, coord in zip(atoms.get_chemical_symbols(), xyz, strict=False):
                    f.write(f"{atom} {float(coord[0]):.8f} {float(coord[1]):.8f} {float(coord[2]):.8f}\n")
        energy_str = np.array2string(energy, precision=1, floatmode="fixed")
        logging.info(f"ITERATION: {self.iteration} DIST: {self.dist:.2f} ENERGY: {energy_str}")

        if not self.growing:
            with gradfile.open("w") as f:
                f.write(f"{self.ngrad}\n")
