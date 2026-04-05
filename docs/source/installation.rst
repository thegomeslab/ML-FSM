Installation
============

Requirements
------------

- Python 3.11+
- `NumPy <https://numpy.org>`_ ≥ 1.26
- `ASE <https://wiki.fysik.dtu.dk/ase>`_ ≥ 3.22
- `SciPy <https://scipy.org>`_ ≥ 1.13
- `geomeTRIC <https://github.com/leeping/geomeTRIC>`_ ≥ 1.0.0
- `NetworkX <https://networkx.org>`_ ≥ 3.0
- `RDKit <https://www.rdkit.org>`_ ≥ 2022.09

Install from PyPI
-----------------

.. code-block:: bash

   pip install mlfsm

Install from Source
-------------------
To access the most recent features and bug fixes, install from source:

.. code-block:: bash

   git clone https://github.com/thegomeslab/ML-FSM.git
   cd ML-FSM
   pip install -e .

Optional: ML Calculator Dependencies
--------------------------------------

ML-FSM works with any ASE-compatible calculator. The following ML potentials
have been tested and are supported out of the box via ``examples/fsm_example.py``:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Calculator
     - Install
     - Notes
   * - **AIMNet2**
     - .. code-block:: bash

          git clone https://github.com/isayevlab/AIMNet2.git
          cd AIMNET2
          pip install -e .
     - AIMNet2 pre-trained model.
   * - **MACE**
     - ``pip install mace-torch``
     - Pre-trained models available via ``mace_off``
   * - **FAIR UMA**
     - ``pip install fairchem-core``
     - Meta's Universal Model for Atoms
   * - **xTB**
     - ``mamba install xtb-python``
     - GFN2-xTB semiempirical method
   * - **Q-Chem**
     - See `Q-Chem <https://www.q-chem.com>`_
     - File-based I/O; requires a Q-Chem license
