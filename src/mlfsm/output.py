"""Incremental output file writer for FSM calculations."""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TextIO

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlfsm.cos import FreezingString

_SEP = "=" * 70
_SEP_THIN = "-" * 70


def _write_section(f: TextIO, title: str) -> None:
    f.write(f"\n {_SEP}\n")
    pad = (68 - len(title)) // 2
    f.write(f" {' ' * pad}{title}\n")
    f.write(f" {_SEP}\n")


def _format_atoms_block(atoms: Atoms, indent: str = "   ") -> str:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    lines = [
        f"{indent}{i + 1:4d}  {sym:<3s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}"
        for i, (sym, pos) in enumerate(zip(symbols, positions, strict=True))
    ]
    return "\n".join(lines)


def _chemical_formula(atoms: Atoms) -> str:
    counts: Counter[str] = Counter(atoms.get_chemical_symbols())
    order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
    parts = []
    for sym in order:
        if sym in counts:
            parts.append(sym if counts[sym] == 1 else f"{sym}{counts[sym]}")
    return "".join(parts)


def get_calculator_info(calc: Any) -> dict[str, Any]:
    """Extract available information from an ASE calculator without raising."""
    info: dict[str, Any] = {"name": type(calc).__name__}

    for attr in ("label", "method", "basis", "charge", "multiplicity"):
        try:
            val = getattr(calc, attr, None)
            if val is not None:
                info[attr] = val
        except Exception:
            pass

    try:
        params = calc.parameters
        if isinstance(params, dict):
            for k, v in params.items():
                if k not in info:
                    info[k] = v
    except Exception:
        pass

    try:
        d = calc.todict()
        if isinstance(d, dict):
            for k, v in d.items():
                if k not in info:
                    info[k] = v
    except Exception:
        pass

    # FAIRChem / UMA
    for attr in ("task_name",):
        try:
            val = getattr(calc, attr, None)
            if val is not None:
                info[attr] = val
        except Exception:
            pass
    try:
        ckpt = calc.predictor.checkpoint_path  # type: ignore[union-attr]
        info["checkpoint"] = str(ckpt)
    except Exception:
        pass

    return info


class FSMOutput:
    """Manages incremental writing of a human-readable FSM output file.

    Parameters
    ----------
    outdir : path-like
        Directory in which to write the output file.
    filename : str, optional
        Output file name. Default is ``"fsm.out"``.
    """

    def __init__(self, outdir: Path | str, filename: str = "fsm.out") -> None:
        self._path = Path(outdir) / filename
        self._f: TextIO = self._path.open("w", encoding="utf-8")
        self._current_iteration: int = 0
        self._node_lines: list[str] = []

    def close(self) -> None:
        """Flush and close the output file."""
        self._f.flush()
        self._f.close()

    # ------------------------------------------------------------------
    # Setup sections (called once before the main loop)
    # ------------------------------------------------------------------

    def write_header(self, version: str) -> None:
        """Write the banner and timestamp."""
        f = self._f
        f.write(f" {_SEP}\n")
        title = "ML-FSM: Machine Learning Freezing String Method"
        pad = (68 - len(title)) // 2
        f.write(f" {' ' * pad}{title}\n")
        ver_line = f"Version {version}"
        pad2 = (68 - len(ver_line)) // 2
        f.write(f" {' ' * pad2}{ver_line}\n")
        f.write(f" {_SEP}\n")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n Date/Time: {now}\n")
        f.flush()

    def write_parameters(
        self,
        optcoords: str,
        interp: str,
        method: str,
        maxiter: int,
        maxls: int,
        dmax: float,
        nnodes_min: int,
        ninterp: int,
        stepsize: float,
    ) -> None:
        """Write the input parameter block."""
        f = self._f
        _write_section(f, "INPUT PARAMETERS")
        f.write(f"\n   Optimization coordinates   : {optcoords}\n")
        f.write(f"   Interpolation method       : {interp}\n")
        f.write(f"   Optimizer                  : {method}\n")
        f.write(f"   Max optimizer iterations   : {maxiter}\n")
        f.write(f"   Max line search iterations : {maxls}\n")
        f.write(f"   Max displacement (dmax)    : {dmax:.4f} Å\n")
        f.write(f"   Target node count          : {nnodes_min}\n")
        f.write(f"   Interpolation points       : {ninterp}\n")
        if stepsize > 0.0:
            f.write(f"   Step size (explicit)       : {stepsize:.4f} Å\n")
        else:
            f.write(f"   Step size (explicit)       : derived from target node count\n")
        f.flush()

    def write_system_info(
        self,
        reactant: Atoms,
        product: Atoms,
        chg: int,
        mult: int,
        fixed_atoms: Optional[NDArray[np.integer[Any]]],
    ) -> None:
        """Write molecular system information."""
        f = self._f
        _write_section(f, "MOLECULAR SYSTEM")
        formula = _chemical_formula(reactant)
        natoms = len(reactant)
        f.write(f"\n   Formula       : {formula}\n")
        f.write(f"   Atoms         : {natoms}\n")
        f.write(f"   Charge        : {chg}\n")
        f.write(f"   Multiplicity  : {mult}\n")
        if fixed_atoms is None or len(fixed_atoms) == 0:
            f.write("   Fixed atoms   : None\n")
        else:
            idx_str = ", ".join(str(i + 1) for i in fixed_atoms)
            f.write(f"   Fixed atoms   : {idx_str} (1-indexed)\n")
        f.flush()

    def write_calculator_info(self, calc: Any) -> None:
        """Write calculator name and available parameters."""
        f = self._f
        info = get_calculator_info(calc)
        _write_section(f, "CALCULATOR")
        f.write(f"\n   Calculator    : {info.pop('name')}\n")
        skip = {"kwargs", "restart", "ignore_bad_restart_file", "directory"}
        for k, v in info.items():
            if k in skip:
                continue
            label = k.replace("_", " ").capitalize()
            f.write(f"   {label:<20s}: {v}\n")
        f.flush()

    def write_initial_structures(self, reactant: Atoms, product: Atoms) -> None:
        """Write reactant and product coordinate blocks."""
        f = self._f
        _write_section(f, "INITIAL STRUCTURES")
        f.write("\n   Standard Orientation — Reactant (Angstroms)\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {'Idx':>4s}  {'Sym':<3s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(_format_atoms_block(reactant))
        f.write("\n")

        f.write(f"\n   Standard Orientation — Product (Angstroms)\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {'Idx':>4s}  {'Sym':<3s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(_format_atoms_block(product))
        f.write("\n")
        f.flush()

    def write_path_init(self, dist: float, stepsize: float, nnodes_min: int) -> None:
        """Write path initialization summary."""
        f = self._f
        _write_section(f, "PATH INITIALIZATION")
        f.write(f"\n   Total path distance   : {dist:.4f} Å\n")
        f.write(f"   Step size             : {stepsize:.4f} Å\n")
        f.write(f"   Target node count     : {nnodes_min}\n")
        f.flush()

    # ------------------------------------------------------------------
    # Per-iteration sections (called from cos.py hooks)
    # ------------------------------------------------------------------

    def _ensure_iteration_header(self, iteration: int) -> None:
        if iteration != self._current_iteration:
            self._current_iteration = iteration
            self._node_lines = []
            _write_section(self._f, f"ITERATION {iteration}")
            self._f.write("\n")

    def write_frontier_node(self, side: str, atoms: Atoms, dist: float) -> None:
        """Write the newly selected frontier node structure.

        Parameters
        ----------
        side : {"r", "p"}
            Which end of the string.
        atoms : Atoms
            The frontier node geometry.
        dist : float
            Current distance between frontier nodes.
        """
        f = self._f
        label = "Reactant-side frontier" if side == "r" else "Product-side frontier"
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {label}  (frontier distance = {dist:.4f} Å)\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {'Idx':>4s}  {'Sym':<3s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}\n")
        f.write(_format_atoms_block(atoms))
        f.write("\n")
        f.flush()

    def write_optimized_node(
        self,
        side: str,
        idx: int,
        atoms: Atoms,
        energy: Optional[float],
        ngrad: int,
    ) -> None:
        """Record an optimized (or endpoint-evaluated) node.

        Called from within ``FreezingString.optimize()``.  Results are
        buffered and flushed together with the iteration summary.

        Parameters
        ----------
        side : {"r", "p"}
            Which string the node belongs to.
        idx : int
            Position in the string list (0 = endpoint).
        atoms : Atoms
            Final geometry after optimization.
        energy : float or None
            Energy in eV.
        ngrad : int
            Number of gradient calls used (0 for endpoint-only evaluation).
        """
        tag = f"{side}[{idx}]"
        kind = "endpoint " if ngrad == 0 else "optimized"
        if energy is not None:
            e_str = f"{energy:+.6f} eV"
        else:
            e_str = "N/A"
        grad_str = "" if ngrad == 0 else f"   ngrad = {ngrad}"
        self._node_lines.append(f"     {tag:<8s}  {kind}  :  energy = {e_str}{grad_str}")

        f = self._f
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   Optimized node {tag} ({kind})\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   Energy: {e_str}{grad_str}\n")
        f.write(f"   {'Idx':>4s}  {'Sym':<3s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}\n")
        f.write(_format_atoms_block(atoms))
        f.write("\n")
        f.flush()

    def write_iteration_summary(
        self,
        iteration: int,
        r_energies: list[Optional[float]],
        p_energies: list[Optional[float]],
        dist: float,
    ) -> None:
        """Write per-iteration energy table and distance.

        Called from ``FreezingString.write()`` after the XYZ file is written.
        """
        f = self._f
        all_energies = r_energies + p_energies[::-1]
        valid = [e for e in all_energies if e is not None]
        if not valid:
            return
        e_min = min(valid)

        f.write(f"\n   {_SEP_THIN}\n")
        f.write(f"   Iteration {iteration} summary   (frontier distance = {dist:.4f} Å)\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {'Node':<8s}  {'Side':<8s}  {'Energy (eV)':>14s}  {'Rel. Energy (eV)':>18s}\n")
        f.write(f"   {_SEP_THIN}\n")

        nr = len(r_energies)
        for i, e in enumerate(r_energies):
            tag = "R" if i == 0 else ""
            e_str = f"{e:+.6f}" if e is not None else "  N/A   "
            rel_str = f"{e - e_min:+.4f}" if e is not None else "  N/A "
            f.write(f"   {i + 1:<8d}  {'r' + tag:<8s}  {e_str:>14s}  {rel_str:>18s}\n")

        for j, e in enumerate(p_energies[::-1]):
            node_idx = nr + j + 1
            tag = "P" if j == len(p_energies) - 1 else ""
            e_str = f"{e:+.6f}" if e is not None else "  N/A   "
            rel_str = f"{e - e_min:+.4f}" if e is not None else "  N/A "
            f.write(f"   {node_idx:<8d}  {'p' + tag:<8s}  {e_str:>14s}  {rel_str:>18s}\n")

        f.write(f"   {_SEP_THIN}\n\n")
        f.flush()

    # ------------------------------------------------------------------
    # Final summary (called once after the loop)
    # ------------------------------------------------------------------

    def write_final_summary(self, string: "FreezingString") -> None:
        """Write TS guess identification and full string energy profile."""
        f = self._f
        _write_section(f, "CALCULATION COMPLETE")

        all_energies = string.r_energy + string.p_string[::-1]  # type: ignore[operator]
        all_energies = string.r_energy + string.p_energy[::-1]
        valid_pairs = [(i, e) for i, e in enumerate(all_energies) if e is not None]

        f.write(f"\n   Total iterations       : {string.iteration}\n")
        f.write(f"   Total gradient calls   : {string.ngrad}\n\n")

        if not valid_pairs:
            f.write("   No energies available.\n")
            f.flush()
            return

        e_values = np.array([e for _, e in valid_pairs])
        e_min = float(e_values.min())
        ts_local = int(np.argmax(e_values))
        ts_global_idx = valid_pairs[ts_local][0]
        ts_energy = valid_pairs[ts_local][1]
        assert ts_energy is not None

        nr = len(string.r_string)
        if ts_global_idx < nr:
            ts_label = f"r[{ts_global_idx}]"
        else:
            p_idx = len(all_energies) - 1 - ts_global_idx
            ts_label = f"p[{p_idx}]"

        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   TS Guess: node {ts_label}  (highest-energy node)\n")
        f.write(f"     Absolute energy  : {ts_energy:+.6f} eV\n")
        f.write(f"     Relative energy  : {ts_energy - e_min:+.4f} eV  (above string minimum)\n")
        f.write(f"   {_SEP_THIN}\n\n")

        f.write(f"   Full String Energies (relative to minimum, eV)\n")
        f.write(f"   {_SEP_THIN}\n")
        f.write(f"   {'Node':<8s}  {'Side':<8s}  {'Energy (eV)':>14s}  {'Rel. Energy (eV)':>18s}\n")
        f.write(f"   {_SEP_THIN}\n")

        for local_i, (global_i, e) in enumerate(valid_pairs):
            if global_i < nr:
                side = "R" if global_i == 0 else "r"
            else:
                p_pos = len(all_energies) - 1 - global_i
                side = "P" if p_pos == 0 else "p"
            ts_marker = "  <-- TS guess" if local_i == ts_local else ""
            e_str = f"{e:+.6f}"
            rel_str = f"{e - e_min:+.4f}"
            f.write(
                f"   {local_i + 1:<8d}  {side:<8s}  {e_str:>14s}  {rel_str:>18s}{ts_marker}\n"
            )

        f.write(f"   {_SEP_THIN}\n")
        f.write(f"\n   Full string written to: vfile_{string.iteration:02d}.xyz\n")
        f.write(f"\n {_SEP}\n")
        f.flush()
