"""Unit tests for the golden-comparison helpers themselves."""

from __future__ import annotations

from tests.golden_utils import compare_lines, normalize_output


def test_normalize_masks_datetime() -> None:
    text = " Date/Time: 2026-06-17 15:44:01\n other line\n"
    out = normalize_output(text)
    assert out == " Date/Time: <MASKED>\n other line"


def test_normalize_leaves_other_lines_untouched() -> None:
    text = "   C    1.000000   2.000000   3.000000"
    assert normalize_output(text) == text


def test_normalize_is_idempotent() -> None:
    text = " Date/Time: 2026-06-17 15:44:01\n   C  1.0  2.0  3.0"
    once = normalize_output(text)
    assert normalize_output(once) == once


def test_compare_lines_matches_within_tolerance() -> None:
    golden = "   C    1.000000   2.000000   energy +1.234567 eV"
    actual = "   C    1.000004   1.999996   energy +1.234561 eV"
    assert compare_lines(golden, actual, abs_tol=1e-5) == []


def test_compare_lines_flags_numeric_outside_tolerance() -> None:
    golden = "   energy +1.234567 eV"
    actual = "   energy +1.300000 eV"
    diffs = compare_lines(golden, actual, abs_tol=1e-5)
    assert len(diffs) == 1
    assert "numeric mismatch" in diffs[0]


def test_compare_lines_flags_text_mismatch() -> None:
    diffs = compare_lines("   energy +1.0 eV", "   energy +1.0 kcal")
    assert len(diffs) == 1
    assert "text mismatch" in diffs[0]


def test_compare_lines_flags_token_count_change() -> None:
    diffs = compare_lines("a b c", "a b")
    assert any("token count differs" in d for d in diffs)


def test_compare_lines_flags_line_count_change() -> None:
    diffs = compare_lines("a\nb", "a")
    assert any("line count differs" in d for d in diffs)
