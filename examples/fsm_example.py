# ruff: noqa: PLC0415
"""
Example script for running the Freezing String Method (FSM).

Users must install their desired quantum chemistry backend separately from the
mlfsm package. Currently supported calculators include:

    - QChem
    - xTB (GFN2-xTB)
    - FAIR UMA
    - AIMNet2
    - MACEOFF
    - EMT

Only the selected calculator needs to be installed in the Python environment.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer, InternalsOptimizer, Optimizer
from mlfsm.utils import load_xyz, load_xyz_fixed

HERE = Path(__file__).parent


def run_fsm(
    reaction_dir: Path | str,
    optcoords: str = "cart",
    interp: str = "lst",
    method: str = "L-BFGS-B",
    maxls: int = 3,
    maxiter: int = 1,
    dmax: float = 0.3,
    nnodes_min: int = 10,
    stepsize: float = 0.0,
    ninterp: int = 100,
    suffix: str | None = None,
    calculator: str = "qchem",
    fixed: str = "",
    chg: int = 0,
    mult: int = 1,
    nt: int = 1,
    verbose: bool = False,
    ckpt: Path = HERE / "pre_trained_gnns/schnet_fine_tuned.ckpt",
    interpolate: bool = False,
    **kwargs,
):
    """Run the Freezing String Method on a given reaction with user specified parameters."""
    reaction_dir = Path(reaction_dir)

    if suffix:
        outdir = reaction_dir / (
            f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_"
            f"maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}_{suffix}"
        )
    else:
        outdir = (
            reaction_dir
            / f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}"
        )

    if interpolate:
        outdir = reaction_dir / f"interp_{interp}"

    if outdir.is_dir():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # get fixed atom indices
    def parse_indices(text):
        if text is None or text.strip() == "":
            return np.array([], dtype=int)
        indices = []
        for part_raw in text.split(","):
            part = part_raw.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        return np.array(indices, dtype=int) - 1

    fixed_atoms = parse_indices(fixed)

    # Load structures
    if len(fixed) > 0:
        reactant, product = load_xyz_fixed(reaction_dir, fixed=fixed_atoms)
    else:
        reactant, product = load_xyz(reaction_dir)
    with open(os.path.join(reaction_dir, "chg")) as f:
        chg = int(f.read())
    with open(os.path.join(reaction_dir, "mult")) as f:
        mult = int(f.read())

    calc: Any

    # works for FAIRchem models and Q-Chem
    reactant.info.update({"charge": chg, "spin": mult})
    product.info.update({"charge": chg, "spin": mult})

    # some models prefer to have it set in the initial_charges
    chg_list = reactant.get_initial_charges()
    chg_list[0] = chg
    mult_list = reactant.get_initial_magnetic_moments()
    mult_list[0] = mult - 1

    reactant.set_initial_magnetic_moments(mult_list)
    reactant.set_initial_charges(chg_list)

    product.set_initial_magnetic_moments(mult_list)
    product.set_initial_charges(chg_list)

    # Load calculator
    if calculator == "qchem":
        from ase.calculators.qchem import QChem

        calc = QChem(
            label="fsm",
            method="wb97x-v",
            basis="def2-tzvp",
            charge=chg,
            multiplicity=mult,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=nt,
        )
    elif calculator == "xtb":
        from xtb.ase.calculator import XTB  # type: ignore [import-not-found]

        calc = XTB(method="GFN2-xTB")
    elif calculator == "uma":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
    elif calculator == "torchmd":
        from custom_calculator_torchmd import TMDCalculator

        calc = TMDCalculator()
    elif calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE  # type: ignore [import-not-found]

        calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
    elif calculator == "emt":
        from ase.calculators.emt import EMT

        calc = EMT()
    elif calculator == "mace":
        import torch
        from mace.calculators import mace_off  # type: ignore [import-not-found]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(model="large", device=dev)
    else:
        raise ValueError(f"Unknown calculator {calculator}")

    # Initialize FSM string
    string = FreezingString(reactant, product, nnodes_min, interp, ninterp, stepsize)
    if interpolate:
        string.interpolate(outdir)
        return

    optimizer: Optimizer
    # Choose optimizer
    if optcoords == "cart":
        optimizer = CartesianOptimizer(calc, method, maxiter, maxls, dmax)
    elif optcoords == "ric":
        optimizer = InternalsOptimizer(calc, method, maxiter, maxls, dmax)
    else:
        raise ValueError("Check optimizer coordinates")

    # Run FSM
    while string.growing:
        string.grow()
        string.optimize(optimizer)
        string.write(outdir)

    print(f"Gradient calls: {string.ngrad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction_dir", type=Path, help="absolute path to reaction")
    parser.add_argument(
        "--optcoords", type=str, default="cart", choices=["cart", "ric"], help="Coordinates for optimization"
    )
    parser.add_argument(
        "--interp", type=str, default="ric", choices=["cart", "lst", "ric"], help="Interpolation method"
    )
    parser.add_argument("--nnodes_min", type=int, default=18, help="Minimum number of nodes in the FSM string")
    parser.add_argument(
        "--stepsize",
        type=float,
        default=0.0,
        help="Stepsize in Angstrom used in interpolation. Overrides and sets nnodes_min based on Cartesian distance.",
    )
    parser.add_argument("--ninterp", type=int, default=50, help="Number of interpolation points between nodes")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for output directory")
    parser.add_argument(
        "--method", type=str, default="L-BFGS-B", choices=["L-BFGS-B", "CG"], help="Optimization method"
    )
    parser.add_argument("--maxls", type=int, default=3, help="Maximum number of line search iterations")
    parser.add_argument("--maxiter", type=int, default=2, help="Maximum number of optimization iterations")
    parser.add_argument("--dmax", type=float, default=0.05, help="Maximum displacement for optimization steps")
    parser.add_argument(
        "--calculator",
        type=str,
        default="schnet",
        choices=["qchem", "xtb", "schnet", "torchmd", "uma", "aimnet2", "emt"],
        help="Calculator to use for energy and gradient evaluations",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=HERE / "pre_trained_gnns/schnet_fine_tuned.ckpt",
        help="Checkpoint for calculator",
    )
    parser.add_argument(
        "--fixed", type=str, default="", help="Fix atoms, 1-indexed. usage: 1-13 fixes the first 12 atoms"
    )
    parser.add_argument("--chg", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--nt", type=int, default=1, help="Number of threads for the calculator")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--interpolate", action="store_true", help="Run interpolation instead of FSM")

    args = parser.parse_args()
    run_fsm(**vars(args))
