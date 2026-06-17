"""Unit tests for the FSM output helpers in mlfsm.output."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.emt import EMT

from mlfsm.output import _chemical_formula, get_calculator_info


def test_chemical_formula_orders_and_counts() -> None:
    # C first, H second, remaining elements alphabetical; single counts suppressed.
    assert _chemical_formula(Atoms("OCHN")) == "CHNO"
    assert _chemical_formula(Atoms("CCHHHO")) == "C2H3O"


def test_get_calculator_info_emt() -> None:
    assert get_calculator_info(EMT())["name"] == "EMT"
