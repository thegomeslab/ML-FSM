Examples
========

Quick Start — EMT Calculator
-----------------------------

The simplest way to test ML-FSM is with ASE's built-in EMT calculator.
The example below runs a full FSM calculation on the Diels-Alder reaction
included in the ``examples/data/`` directory. Note that the EMT calculator
is only for quick tests and demonstrations and should not be used in
serious TS searches.

.. code-block:: python

    from pathlib import Path
    from ase.calculators.emt import EMT
    from mlfsm.utils import load_xyz
    from mlfsm.cos import FreezingString
    from mlfsm.opt import CartesianOptimizer

    reaction_dir = Path("examples/data/06_diels_alder")
    reactant, product = load_xyz(reaction_dir)

    # Set up the FSM with RIC interpolation (recommended)
    fsm = FreezingString(reactant, product, nnodes_min=9, interp_method="ric")

    # Set up the optimizer with an ASE calculator
    optimizer = CartesianOptimizer(calc=EMT(), maxiter=1, maxls=3)

    outdir = Path("fsm_output")
    outdir.mkdir(exist_ok=True)

    # Run the FSM loop
    while fsm.growing:
        fsm.grow()
        fsm.optimize(optimizer)
        fsm.write(outdir)

    print(f"FSM finished in {fsm.ngrad} gradient evaluations")
    print(f"TS guess written to {outdir}")

Using Fixed Atoms
-----------------

To freeze part of the system (e.g., a scaffold or periodic image atoms), pass
a zero-indexed array of atom indices. Fixed atoms are excluded from interpolation
displacement and held stationary during node optimization.

.. code-block:: python

    import numpy as np
    from mlfsm.utils import load_xyz_fixed
    from mlfsm.cos import FreezingString

    # Freeze atoms 0–11 (zero-indexed)
    fixed = np.arange(0, 12)
    reactant, product = load_xyz_fixed("examples/data/my_reaction", fixed)

    fsm = FreezingString(reactant, product, nnodes_min=9, interp_method="ric")

The ``FixAtoms`` constraint is propagated automatically through the FSM;
:func:`~mlfsm.geom.project_trans_rot_fixed` is used for alignment and
:class:`~mlfsm.opt.CartesianOptimizer` zeroes gradients on fixed atoms.

Specifying a Step Size Directly
---------------------------------

Instead of setting ``nnodes_min``, you can pass an explicit Cartesian step
size (in Angstrom). The number of nodes is then determined automatically from
the path arc length.

.. code-block:: python

    fsm = FreezingString(reactant, product, stepsize=0.3, interp_method="ric")

Choosing an Interpolation Method
----------------------------------

Three interpolation methods are available via the ``interp_method`` argument:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``interp_method``
     - Description
   * - ``"ric"`` *(default)*
     - Linear interpolation in redundant internal coordinates. Produces
       physically realistic intermediate geometries and generally gives the
       best TS guess quality.
   * - ``"lst"``
     - Linear Synchronous Transit. Interpolates pairwise interatomic distances
       and minimises deviation from a Cartesian linear blend.
   * - ``"cart"``
     - Plain Cartesian linear interpolation. Fast but can produce unphysical
       structures for large conformational changes.

Using an ML Calculator (UMA Small 1.2)
--------------------------------------

ML-FSM is calculator-agnostic — any ASE-compatible calculator can be used as
the gradient provider. Below is a minimal example using the UMA Small 1.2 model.

.. code-block:: python

    from pathlib import Path
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from mlfsm.utils import load_xyz
    from mlfsm.cos import FreezingString
    from mlfsm.opt import CartesianOptimizer

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    reactant, product = load_xyz("examples/data/06_diels_alder")
    fsm = FreezingString(reactant, product, nnodes_min=9, interp_method="ric")
    optimizer = CartesianOptimizer(calc=calc, maxiter=1, maxls=3)

    outdir = Path("fsm_output")
    outdir.mkdir(exist_ok=True)

    while fsm.growing:
        fsm.grow()
        fsm.optimize(optimizer)
        fsm.write(outdir)

Google Colab Tutorial
---------------------
See ``examples/FSM_Colab_AIMNet2.ipynb`` for an Colab notebook using AIMNet2.

Example Reactions
-----------------

The ``examples/data/`` directory contains nine test reactions:

.. list-table::
   :header-rows: 1
   :widths: 10 55

   * - Folder
     - Reaction
   * - ``01_formaldehyde``
     - H₂CO → H₂ + CO
   * - ``02_isocyanate_water``
     - Isocyanate + H₂O → carbamic acid
   * - ``03_ethanal``
     - Acetaldehyde keto–enol tautomerism
   * - ``04_ethane_dehydrogenation``
     - CH₃CH₃ → CH₂CH₂ + H₂
   * - ``05_bicyclobutane``
     - Bicyclo[1.1.0]butane → *trans*-butadiene
   * - ``06_diels_alder``
     - Parent Diels–Alder cycloaddition
   * - ``07_hexadiene``
     - *cis,cis*-2,4-hexadiene → 3,4-dimethylcyclobutene
   * - ``08_alanine``
     - Alanine dipeptide rearrangement
   * - ``09_proton``
     - Acid zeolite proton migration

Each directory contains ``initial.xyz`` (reactant + product frames), ``chg``
(molecular charge), and ``mult`` (spin multiplicity).
