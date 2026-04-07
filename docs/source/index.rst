ML-FSM Documentation
====================

ML-FSM is a Python implementation of the **Freezing String Method** (FSM) for automated
transition state (TS) guess generation. It supports multiple interpolation strategies
including redundant internal coordinates (RIC), linear synchronous transit (LST), and
Cartesian interpolation, and is designed to work with any ASE-compatible calculator,
including machine learning interatomic potentials.

Key features:

- **RIC interpolation** — produces chemically realistic intermediate geometries and
  enables larger interpolation step sizes with fewer optimization steps per cycle
- **Calculator agnostic** — works with ML potentials (AIMNet2, MACE-OFF, TensorNet,
  xTB, FAIR UMA) and quantum chemistry codes (Q-Chem via file I/O)
- **Fixed-atom support** — freeze a subset of atoms during interpolation and optimization
- **Flexible step-size control** — specify node count or an explicit Cartesian step size
- **Open source** — MIT-licensed, installable from PyPI

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
