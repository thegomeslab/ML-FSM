API Reference
=============

This section contains the complete API reference for all modules in ``mlfsm``.

.. contents:: Modules
   :local:
   :depth: 1

----

Core Modules
------------

FreezingString (``mlfsm.cos``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main driver class that orchestrates string growth, interpolation, and node
optimization throughout the FSM calculation.

.. automodule:: mlfsm.cos
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Interpolation (``mlfsm.interp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides :class:`~mlfsm.interp.Linear`, :class:`~mlfsm.interp.LST`, and
:class:`~mlfsm.interp.RIC` interpolation schemes for generating reaction path
nodes between endpoint geometries.

.. automodule:: mlfsm.interp
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Coordinates (``mlfsm.coords``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cartesian and redundant internal coordinate systems used during interpolation and
back-transformation to Cartesian coordinates.

.. automodule:: mlfsm.coords
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Optimization (``mlfsm.opt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Node-level optimizers that relax FSM images perpendicular to the current tangent
direction using L-BFGS-B with explicit line search.

.. automodule:: mlfsm.opt
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Geometry Utilities (``mlfsm.geom``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-level vector operations and projection operators used throughout the FSM,
including rigid-body alignment and translation/rotation projection.

.. automodule:: mlfsm.geom
   :members:
   :undoc-members:
   :show-inheritance:

----

Atom Mapping (``mlfsm.atom_mapping``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determines a consistent atom ordering between reactant and product using maximum
common substructure (MCS) matching followed by Hungarian assignment for
bond-breaking/forming atoms.

.. automodule:: mlfsm.atom_mapping
   :members:
   :undoc-members:
   :show-inheritance:

----

Utilities (``mlfsm.utils``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input/output helpers for loading reactant and product geometries and handling
fixed-atom constraints.

.. automodule:: mlfsm.utils
   :members:
   :undoc-members:
   :show-inheritance:
