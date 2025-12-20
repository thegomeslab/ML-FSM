"""Coordinate generation and transformation tools for FSM optimization."""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, vdw_radii
from ase.units import Bohr
from geometric.internal import (  # type: ignore [import-untyped]
    Angle,
    CartesianX,
    CartesianY,
    CartesianZ,
    Dihedral,
    Distance,
    LinearAngle,
    OutOfPlane,
)
from numpy.typing import NDArray

angs_to_bohr = 1 / Bohr
deg_to_rad = np.pi / 180.0
MIN_ATOMS_FOR_TORSION = 4
RMS_DX_THRESHOLD = 1e-7
MAX_ITERATIONS = 200
EIGENVAL_CUTOFF = 1e-8


class Coordinates:
    """Base class for internal coordinate systems used in FSM."""

    def __init__(self, atoms1: Atoms, atoms2: Optional[Atoms] = None, verbose: bool = False) -> None:
        self.atoms1 = atoms1
        self.atoms2 = atoms2
        c = atoms1.constraints  # constraint indicies must be identical between R&P therefor only one is needed
        if len(c) > 0:
            self.fixed_atoms = c[0].get_indices()
        else:
            self.fixed_atoms = np.array([])
        self.coords = self.construct()
        self.keys = list(self.coords.keys())
        self.verbose = verbose
        if self.atoms2 is not None and verbose:
            self.dqprint(self.atoms1, self.atoms2)
        elif verbose:
            self.qprint(self.atoms1)

    def construct(self) -> Dict[str, Any]:
        """Construct the coordinate representation for a given atom set."""
        raise NotImplementedError("No construct function")

    def qprint(self, atoms: Atoms) -> None:
        """Print coordinate values in human-readable format for a given ASE atoms object."""
        xyz = atoms.get_positions()
        xyzb = xyz * angs_to_bohr
        print(f"\n{'Coordinate':15}{'Value':15}")
        for name, coord in self.coords.items():
            print(f"{name:15} = {coord.value(xyzb):15.8f}")

    def q(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return coordinate values in from Cartesian positions."""
        xyzb = xyz * angs_to_bohr
        # return np.array([coord.value(xyzb) for coord in self.coords.values()], dtype=np.float64)
        return np.fromiter((coord.value(xyzb) for coord in self.coords.values()), dtype=np.float64)

    def dqprint(self, atoms1: Atoms, atoms2: Atoms) -> None:
        """Print differences in internal coordinates between two structures."""
        q1 = self.q(atoms1.get_positions())
        q2 = self.q(atoms2.get_positions())
        print(f"\n{'Coordinate':15}{'Value':15}")
        for name, q1_i, q2_i, dq_i in zip(self.keys, q1, q2, (q2 - q1), strict=True):
            star = ""
            if ("bend" in name or "tors" in name or "oop" in name) and dq_i < -np.pi:
                star = "*"
            elif ("bend" in name or "tors" in name or "oop" in name) and dq_i > np.pi:
                star = "*"
            print(f"{name:15s} = {q1_i:15.8f} {q2_i:15.8f} {dq_i:15.8f} {star}")

    def b_matrix(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Construct the B-matrix for internal coordinates."""
        xyzb = xyz * angs_to_bohr
        nint = len(self.coords)
        ncart = xyzb.size
        B = np.zeros((nint, ncart))
        for i, coord in enumerate(self.coords.values()):
            B[i] = coord.derivative(xyzb).flatten()
        return B

    def u_matrix(self, Bprim: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N803
        """Compute projection matrix U from the B-matrix."""
        evals, evecs = np.linalg.eigh(Bprim @ Bprim.T)
        return evecs[:, evals > EIGENVAL_CUTOFF]

    def x(self, xyz: NDArray[np.float64], qtarget: NDArray[np.float64]) -> NDArray[np.float64]:
        """Back-transform internal coordinate displacements to Cartesian updates."""
        xyz1 = xyz.copy()

        for name in self.keys:
            if "linearbnd" in name:
                self.coords[name].reset(xyz1 * angs_to_bohr)

        q0 = self.q(xyz1)
        dq = qtarget - q0
        for i, name in enumerate(self.keys):
            if ("tors" in name) and dq[i] < -np.pi:
                dq[i] += 2 * np.pi
            elif ("tors" in name) and dq[i] > np.pi:
                dq[i] -= 2 * np.pi

        Bprim = self.b_matrix(xyz1)
        U = self.u_matrix(Bprim)
        B = U.T @ Bprim
        BT_inv = np.linalg.pinv(B @ B.T) @ B
        dq = U.T @ dq
        dx = BT_inv.T @ dq
        rms_dx = np.sqrt(np.mean(dx**2))
        rms_dq = np.sqrt(np.mean(dq**2))
        xyz_backup = xyz1.copy() + dx.reshape(-1, 3) / angs_to_bohr
        dq_min = rms_dq

        niter = 1
        while rms_dx > RMS_DX_THRESHOLD:
            xyz1 += dx.reshape(-1, 3) / angs_to_bohr

            q0 = self.q(xyz1)
            dq = qtarget - q0
            for i, name in enumerate(self.keys):
                if ("tors" in name) and dq[i] < -np.pi:
                    dq[i] += 2 * np.pi
                elif ("tors" in name) and dq[i] > np.pi:
                    dq[i] -= 2 * np.pi
            Bprim = self.b_matrix(xyz1)
            U = self.u_matrix(Bprim)
            B = U.T @ Bprim
            BT_inv = np.linalg.pinv(B @ B.T) @ B
            dq = U.T @ dq
            dx = BT_inv.T @ dq
            rms_dx = np.sqrt(np.mean(dx**2))
            rms_dq = np.sqrt(np.mean(dq**2))

            niter += 1
            if niter > MAX_ITERATIONS:
                if self.verbose:
                    print("R FUNCTION FAILED")
                if self.verbose:
                    print(f"Iteration {niter}")
                if self.verbose:
                    print(f"\tRMS(dx) = {rms_dx:10.5e}")
                if self.verbose:
                    print(f"\tRMS(dq) = {rms_dq:10.5e}")

                return np.array(xyz_backup, dtype=np.float64)

            if rms_dq < dq_min:
                xyz_backup = xyz1.copy()

        return xyz1


class Cartesian(Coordinates):
    """Cartesian coordinate system used for atoms."""

    def construct(self) -> Dict[str, Any]:
        """Build Cartesian coordinate representation."""
        coords = {}
        natoms = len(self.atoms1.numbers)
        c = self.atoms1.constraints
        if len(c) > 0:
            fixed_atoms = c[0].get_indices()
        else:
            fixed_atoms = np.array([])

        for i in range(natoms):
            if i not in fixed_atoms:
                coords[f"cartx_{i}"] = CartesianX(i, w=1.0)
                coords[f"carty_{i}"] = CartesianY(i, w=1.0)
                coords[f"cartz_{i}"] = CartesianZ(i, w=1.0)
        return coords


class Redundant(Coordinates):
    """Redundant internal coordinate system including bond, angle, torsion, etc."""

    def checkstre(self, A: NDArray[np.float64], B: NDArray[np.float64], eps: float = 1e-08) -> bool:  # noqa: N803
        """Check if distance between two atoms is significant (non-zero within tolerance)."""
        v0 = A - B
        n = np.maximum(1e-12, v0.dot(v0))
        return n >= eps

    def checkangle(self, A: NDArray[np.float64], B: NDArray[np.float64], C: NDArray[np.float64]) -> bool:  # noqa: N803
        """Check if angle defined by three atoms is physically valid."""
        return self.checkstre(A, B) and self.checkstre(B, C)

    def checktors(
        self,
        A: NDArray[np.float64],  # noqa: N803
        B: NDArray[np.float64],  # noqa: N803
        C: NDArray[np.float64],  # noqa: N803
        D: NDArray[np.float64],  # noqa: N803
    ) -> bool:
        """Check if torsion angle defined by four atoms is physically valid."""
        return self.checkstre(A, B) and self.checkstre(B, C) and self.checkstre(C, D)

    def get_fragments(self, A: NDArray[np.int_]) -> List[NDArray[np.int_]]:  # noqa: N803
        """Return list of fragments as connected components in adjacency matrix."""
        G: nx.Graph = nx.to_networkx_graph(A)
        return [np.array(list(d)) for d in nx.connected_components(G)]

    def connectivity(
        self, atoms: Atoms
    ) -> Tuple[List[NDArray[np.int64]], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Compute connectivity matrices from atomic positions."""
        # this is done in Angstrom
        z = atoms.get_atomic_numbers()
        natoms = len(z)

        # compute covalent bonds
        conn: NDArray[np.int64] = np.zeros((natoms, natoms), dtype=np.int64)
        for i, j in itertools.combinations(range(natoms), 2):
            # R = euclidean(xyz[i], xyz[j])
            R = atoms.get_distance(i, j, mic=True)
            Rcov = covalent_radii[z[i]] + covalent_radii[z[j]]
            if R < 1.3 * Rcov:
                conn[i, j] = np.int64(1)
                conn[j, i] = np.int64(1)

        # find all fragments
        frags = self.get_fragments(conn)
        nfrags = len(frags)

        conn_frag: NDArray[np.int64] = np.zeros((natoms, natoms), dtype=np.int64)
        conn_frag_aux: NDArray[np.int64] = np.zeros((natoms, natoms), dtype=np.int64)
        conn_frag_idx: NDArray[np.int64] = np.zeros((nfrags, nfrags, 2), dtype=np.int64)
        conn_frag_dist: NDArray[np.int64] = np.zeros((nfrags, nfrags), dtype=float)

        # if fragments>1 get interfragment bonds
        if nfrags > 1:
            # find closest interfragment distances
            for i, j in itertools.combinations(range(natoms), 2):
                i_frag: int = int(np.argmax([i in frag for frag in frags]))
                j_frag: int = int(np.argmax([j in frag for frag in frags]))
                if i_frag != j_frag:
                    # check distance
                    conn_frag_ij = conn_frag_dist[i_frag, j_frag]
                    R = atoms.get_distance(i, j, mic=True)
                    # R = euclidean(xyz[i], xyz[j])
                    if conn_frag_ij == 0.0 or conn_frag_ij > R:
                        conn_frag_dist[i_frag, j_frag] = conn_frag_dist[j_frag, i_frag] = R
                        conn_frag_idx[i_frag, j_frag] = np.array([i, j], dtype=np.int64)
                        conn_frag_idx[j_frag, i_frag] = np.array([j, i], dtype=np.int64)

            # set interfrag connectivity matrix
            for i_frag in range(nfrags):
                for j_frag in range(i_frag + 1, nfrags):
                    i, j = conn_frag_idx[i_frag, j_frag]
                    conn_frag[i, j] = np.int64(1)
                    conn_frag[j, i] = np.int64(1)

            # auxillary interfragment bonds are < 2 Ang or < 1.3*interfrag distance
            for i, j in itertools.combinations(range(natoms), 2):
                i_frag: int = int(np.argmax([i in frag for frag in frags]))  # type: ignore[no-redef]
                j_frag: int = int(np.argmax([j in frag for frag in frags]))  # type: ignore[no-redef]
                if i_frag != j_frag:
                    conn_frag_ij = conn_frag_dist[i_frag, j_frag]
                    # R = euclidean(xyz[i], xyz[j])
                    R = atoms.get_distance(i, j, mic=True)
                    if R < 2.0 or R < 1.3 * conn_frag_ij:  # noqa: PLR2004
                        conn_frag_aux[i, j] = np.int64(1)
                        conn_frag_aux[j, i] = np.int64(1)
            conn_frag_aux = conn_frag_aux - conn_frag

        # find hydrogen bond hydrogens
        X_atnum = [7, 8, 9, 15, 16, 17]  # N, O, F, P, S, Cl
        is_hbond_h: NDArray[np.int64] = np.zeros((natoms,), dtype=np.int64)
        for i, j in itertools.combinations(range(natoms), 2):
            if z[i] == 1 and z[j] in X_atnum:
                if conn[i, j]:
                    is_hbond_h[i] = 1
            elif z[j] == 1 and z[i] in X_atnum:
                if conn[i, j]:
                    is_hbond_h[j] = 1

        # find hydrogen bonds
        conn_hbond: NDArray[np.int64] = np.zeros((natoms, natoms), dtype=np.int64)
        for i, j in itertools.combinations(range(natoms), 2):
            if is_hbond_h[i] and not conn[i, j] and z[j] in X_atnum:
                # R = euclidean(xyz[i], xyz[j])
                R = atoms.get_distance(i, j, mic=True)
                Rvdw = vdw_radii[z[i]] + vdw_radii[z[j]]
                if R < 0.9 * Rvdw:
                    conn_hbond[i, j] = conn_hbond[j, i] = 1
            elif is_hbond_h[j] and not conn[i, j] and z[i] in X_atnum:
                # R = euclidean(xyz[i], xyz[j])
                R = atoms.get_distance(i, j, mic=True)
                Rvdw = vdw_radii[z[i]] + vdw_radii[z[j]]
                if R < 0.9 * Rvdw:
                    conn_hbond[i, j] = conn_hbond[j, i] = 1

        return frags, conn, conn_frag, conn_frag_aux, conn_hbond

    def atoms_to_ric(self, atoms: Atoms) -> Dict[str, Any]:
        """Generate a redundant internal coordinate (RIC) set from ASE.Atoms object."""
        angle_thresh = np.cos(175.0 * np.pi / 180.0)

        coords: Dict[str, Any] = {}
        xyz = atoms.get_positions()
        xyzb = xyz * angs_to_bohr
        _frags, conn, conn_frag, conn_frag_aux, conn_hbond = self.connectivity(atoms)
        natoms = len(atoms)

        # remove fixed atoms from connectivity graph to prevent define coords with them
        for i in self.fixed_atoms:
            conn[i, :] = conn[:, i] = 0
            conn_frag[i, :] = conn_frag[:, i] = 0
            conn_frag_aux[i, :] = conn_frag_aux[:, i] = 0
            conn_hbond[i, :] = conn_hbond[:, i] = 0

        total_conn = (conn + conn_frag + conn_hbond) > 0

        # bonds can be: covalent, interfragment, interfragment aux, or hbond
        for i, j in itertools.combinations(range(natoms), 2):
            if total_conn[i, j] or conn_frag_aux[i, j]:
                coords[f"stre_{i}_{j}"] = Distance(i, j)

        # angles can be: covalent, interfragment, or hbond
        for i, j in itertools.permutations(range(natoms), 2):
            if total_conn[i, j]:
                for k in range(i + 1, natoms):
                    if total_conn[j, k]:
                        check = self.checkangle(xyz[i], xyz[j], xyz[k])
                        if not check:
                            continue
                        ang = Angle(i, j, k)
                        if np.cos(ang.value(xyzb)) < angle_thresh:
                            coords[f"linearbnd_{i}_{j}_{k}_0"] = LinearAngle(i, j, k, 0)
                            coords[f"linearbnd_{i}_{j}_{k}_1"] = LinearAngle(i, j, k, 1)
                        else:
                            coords[f"bend_{i}_{j}_{k}"] = ang

        # torsions can be: covalent, interfragment, or hbond
        for i, j in itertools.permutations(range(natoms), 2):
            if total_conn[i, j]:
                for k in range(natoms):
                    if total_conn[j, k] and i != k and j != k:  # noqa: PLR1714
                        for l in range(i + 1, natoms):  # l>i prevents double counting   # noqa: E741
                            if total_conn[k, l] and i != l and j != l and k != l and not total_conn[l, i]:  # noqa: PLR1714
                                check = self.checktors(xyz[i], xyz[j], xyz[k], xyz[l])
                                if not check:
                                    continue
                                ang1 = Angle(i, j, k)
                                ang2 = Angle(j, k, l)
                                if np.abs(np.cos(ang1.value(xyzb))) > np.abs(angle_thresh):
                                    continue
                                if np.abs(np.cos(ang2.value(xyzb))) > np.abs(angle_thresh):
                                    continue
                                coords[f"tors_{i}_{j}_{k}_{l}"] = Dihedral(i, j, k, l)

        # out-of-plane angle
        for b in range(natoms):
            b_neighbors = np.arange(natoms)[total_conn[b] > 0]
            for a in b_neighbors:
                for c in b_neighbors:
                    for d in b_neighbors:
                        if a < c < d:
                            for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):  # noqa: C414
                                ang1 = Angle(b, i, j)
                                ang2 = Angle(i, j, k)
                                if np.abs(np.cos(ang1.value(xyzb))) > np.abs(angle_thresh):
                                    continue
                                if np.abs(np.cos(ang2.value(xyzb))) > np.abs(angle_thresh):
                                    continue
                                if np.abs(np.dot(ang1.normal_vector(xyzb), ang2.normal_vector(xyzb))) > angle_thresh:
                                    coords[f"oop_{b}_{i}_{j}_{k}"] = OutOfPlane(b, i, j, k)
                                    if natoms > 4:  # noqa: PLR2004
                                        break

        return coords

    def construct(self) -> Dict[str, Any]:
        """Construct the full set of internal coordinates based on input atoms."""
        coords1 = self.atoms_to_ric(self.atoms1)
        if self.atoms2 is None:
            return coords1

        coords2 = self.atoms_to_ric(self.atoms2)
        coords = {**coords1, **coords2}

        min_thresh = np.cos(120.0 * np.pi / 180.0)
        angle_thresh = np.cos(175.0 * np.pi / 180.0)
        oop_thresh = np.abs(np.cos(85 * np.pi / 180.0))
        lb_thresh = np.cos(45.0 * np.pi / 180.0)
        tors_thresh = np.abs(np.cos(175.0 * np.pi / 180.0))

        # Check both ends for ill-defined torsions
        keys = list(coords.keys())
        to_delete = []
        to_add: Dict[str, Any] = {}
        xyzb1 = self.atoms1.get_positions() * angs_to_bohr
        xyzb2 = self.atoms2.get_positions() * angs_to_bohr
        for _i, (name, coord) in enumerate(coords.items()):
            if "tors" in name:
                # check angle ABC and angle BCD for both geometries
                a, b, c, d = coord.a, coord.b, coord.c, coord.d
                ang1 = Angle(a, b, c)
                ang2 = Angle(b, c, d)
                if (
                    (np.abs(np.cos(ang1.value(xyzb1))) > tors_thresh)
                    or (np.abs(np.cos(ang1.value(xyzb2))) > tors_thresh)
                    or (np.abs(np.cos(ang2.value(xyzb1))) > tors_thresh)
                    or (np.abs(np.cos(ang2.value(xyzb2))) > tors_thresh)
                ):
                    to_delete.append(name)
                    continue

        for k in set(to_delete):
            del coords[k]

        for name, coord in to_add.items():
            coords[name] = coord

        # Remove angle coordinates displaced greater than pi or oop greater than pi/2
        self.coords = coords
        keys = list(coords.keys())
        to_delete = []
        to_add = {}
        q1 = self.q(self.atoms1.get_positions())
        q2 = self.q(self.atoms2.get_positions())
        for i, (name, coord) in enumerate(coords.items()):
            if ("bend" in name) and (np.cos(q1[i]) < angle_thresh or np.cos(q2[i]) < angle_thresh):
                if np.abs(np.cos(q2[i] - q1[i])) < np.abs(min_thresh):
                    to_delete.append(name)
            if ("oop" in name) and (np.cos(q1[i]) < -oop_thresh or np.cos(q2[i]) < -oop_thresh):
                to_delete.append(name)
            if ("tors" in name) and (np.cos(q1[i]) < -tors_thresh or np.cos(q2[i]) < -tors_thresh):
                to_delete.append(name)
                to_add["stre_{}_{}".format(coord.a, coord.d)] = Distance(coord.a, coord.d)
            if ("linearbnd" in name) and ((np.cos(q1[i]) < lb_thresh) or (np.cos(q2[i]) < lb_thresh)):
                basecoord = name[:-2]
                to_delete.append(basecoord + "_0")
                to_delete.append(basecoord + "_1")
                to_add["bend_{}_{}_{}".format(coord.a, coord.b, coord.c)] = Angle(coord.a, coord.b, coord.c)
            if "linearbnd" in name:
                a, b, c = coord.a, coord.b, coord.c
                ang = Angle(a, b, c)
                angval1 = ang.value(xyzb1)
                angval2 = ang.value(xyzb2)
                if np.abs(np.cos(angval2 - angval1)) > np.abs(min_thresh):
                    basecoord = name[:-2]
                    to_delete.append(basecoord + "_0")
                    to_delete.append(basecoord + "_1")
                    to_add["bend_{}_{}_{}".format(coord.a, coord.b, coord.c)] = Angle(coord.a, coord.b, coord.c)

        for k in set(to_delete):
            del coords[k]

        for name, coord in to_add.items():
            coords[name] = coord

        # remove oop bends containing broken bonding centers
        keys = list(coords.keys())
        to_delete = []
        oop_keys = [i for i in keys if "oop" in i]
        for oopk in oop_keys:
            coord = coords[oopk]
            if not ((oopk in coords1.keys()) and (oopk in coords2.keys())):
                to_delete.append(oopk)
                continue

        for k in set(to_delete):
            del coords[k]

        keys = list(coords.keys())
        natoms = len(self.atoms1)
        tors_keys = [i for i in keys if "tors" in i]
        ntors = len(tors_keys)
        if ntors < 1 and natoms >= MIN_ATOMS_FOR_TORSION:
            xyz1 = self.atoms1.get_positions()
            xyz2 = self.atoms2.get_positions()
            xyzb1 = xyz1 * angs_to_bohr
            xyzb2 = xyz2 * angs_to_bohr

            for i0, j0, k0, l0 in list(itertools.permutations(range(natoms), 4)):
                check1 = self.checktors(xyz1[i0], xyz1[j0], xyz1[k0], xyz1[l0])
                check2 = self.checktors(xyz2[i0], xyz2[j0], xyz2[k0], xyz2[l0])
                check = check1 and check2
                if not check:
                    continue

                unique_perms = [p for p in itertools.permutations([i0, j0, k0, l0], 4) if p[-1] > p[0]]
                for a, b, c, d in unique_perms:
                    ang1 = Angle(a, b, c)
                    ang2 = Angle(b, c, d)
                    tors = Dihedral(a, b, c, d)

                    if np.cos(tors.value(xyzb1)) < -tors_thresh or np.cos(tors.value(xyzb2)) < -tors_thresh:
                        to_add[f"stre_{a}_{d}"] = Distance(a, d)
                        continue
                    if np.abs(np.cos(ang1.value(xyzb1))) > tors_thresh:
                        continue
                    if np.abs(np.cos(ang1.value(xyzb2))) > tors_thresh:
                        continue
                    if np.abs(np.cos(ang2.value(xyzb1))) > tors_thresh:
                        continue
                    if np.abs(np.cos(ang2.value(xyzb2))) > tors_thresh:
                        continue

                    self.coords[f"tors_{a}_{b}_{c}_{d}"] = Dihedral(a, b, c, d)
                    ntors += 1

                if ntors > 0:
                    break

        if ntors == 0 and natoms >= MIN_ATOMS_FOR_TORSION:
            coords = {f"stre_{i}_{j}": Distance(i, j) for i, j in itertools.combinations(range(natoms), 2)}

        return coords
