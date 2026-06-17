"""Unit tests for the FSM output helpers in mlfsm.output."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT

from mlfsm.output import FSMOutput, _chemical_formula, get_calculator_info


def _atoms(symbols: str) -> Atoms:
    """Build a dummy Atoms object (positions are irrelevant for formula tests)."""
    return Atoms(symbols)


def test_chemical_formula_orders_carbon_hydrogen_first() -> None:
    # C first, H second, remaining elements alphabetical.
    assert _chemical_formula(_atoms("OCHN")) == "CHNO"


def test_chemical_formula_suppresses_single_counts() -> None:
    assert _chemical_formula(_atoms("CO")) == "CO"
    assert _chemical_formula(_atoms("CCHHHO")) == "C2H3O"


def test_get_calculator_info_emt() -> None:
    info = get_calculator_info(EMT())
    assert info["name"] == "EMT"


def test_get_calculator_info_never_raises_on_hostile_object() -> None:
    class Hostile:
        @property
        def parameters(self):
            raise RuntimeError("boom")

        def todict(self):
            raise RuntimeError("boom")

    info = get_calculator_info(Hostile())
    assert info["name"] == "Hostile"


def test_write_iteration_summary_handles_all_none(tmp_path: Path) -> None:
    out = FSMOutput(tmp_path)
    # All-None energies hit the early-return branch (never reached in a full run).
    out.write_iteration_summary(1, [None, None], [None], dist=1.0)
    out.close()
    assert "energy summary" not in (tmp_path / "fsm.out").read_text()


def test_write_iteration_summary_renders_na_for_missing(tmp_path: Path) -> None:
    out = FSMOutput(tmp_path)
    out.write_iteration_summary(1, [-1.0, None], [-2.0], dist=1.5)
    out.close()
    text = (tmp_path / "fsm.out").read_text()
    assert "energy summary" in text
    assert "N/A" in text


def test_write_final_summary_no_energies(tmp_path: Path) -> None:
    class FakeString:
        r_energy: list = [None]
        p_energy: list = [None]
        r_string = [Atoms("H")]
        p_string = [Atoms("H")]
        iteration = 0
        ngrad = 0

    out = FSMOutput(tmp_path)
    out.write_final_summary(FakeString())  # type: ignore[arg-type]
    out.close()
    assert "No energies available" in (tmp_path / "fsm.out").read_text()
