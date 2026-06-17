"""Helpers for golden-output regression tests.

These utilities normalize the non-deterministic parts of an ``fsm.out`` file and
perform a line- and token-aware comparison against a stored golden file.

The comparison is numeric-tolerance aware rather than exact byte-for-byte: the
FSM output is byte-identical for repeated runs on a single machine, but the last
printed digit of a coordinate or energy can drift across CPU architectures /
BLAS builds (CI runs on both ``linux-64`` and ``osx-arm64``).  Text and
structure (line count, tokens per line) are still compared exactly, so any real
formatting or output change is caught.
"""

from __future__ import annotations

import math
import re

# The only non-deterministic line in fsm.out is the wall-clock timestamp written
# by FSMOutput.write_header (src/mlfsm/output.py).
_DATETIME_RE = re.compile(r"^( Date/Time: ).*$")
_MASK = r"\1<MASKED>"


def normalize_output(text: str) -> str:
    """Mask the non-deterministic ``Date/Time:`` line in an ``fsm.out`` string.

    Masking (rather than deleting) the line means a regression that *adds* or
    *removes* a timestamp line still changes the structure and is caught by
    :func:`compare_lines`.

    Parameters
    ----------
    text : str
        Raw contents of an ``fsm.out`` file.

    Returns
    -------
    str
        The text with the timestamp value replaced by ``<MASKED>``.
    """
    return "\n".join(_DATETIME_RE.sub(_MASK, line) for line in text.splitlines())


def _try_float(token: str) -> float | None:
    """Return ``token`` as a float, or ``None`` if it is not numeric."""
    try:
        return float(token)
    except ValueError:
        return None


def compare_lines(
    golden: str,
    actual: str,
    *,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-6,
) -> list[str]:
    """Compare two output strings line by line, tolerating last-digit float drift.

    Both strings are split into lines, and each line into whitespace-separated
    tokens.  Tokens that parse as floats are compared with
    :func:`math.isclose`; all other tokens (labels, symbols, units, separators,
    the masked timestamp) are compared exactly.  Line and token counts must match.

    Parameters
    ----------
    golden : str
        The stored, normalized golden output.
    actual : str
        The freshly generated, normalized output.
    abs_tol, rel_tol : float
        Absolute and relative tolerances passed to :func:`math.isclose` for
        numeric tokens.  The default ``abs_tol`` of ``1e-5`` sits just below the
        six-decimal precision of ``fsm.out`` fields.

    Returns
    -------
    list[str]
        Human-readable descriptions of each divergence, with line numbers.  An
        empty list means the two strings match within tolerance.
    """
    g_lines = golden.splitlines()
    a_lines = actual.splitlines()
    diffs: list[str] = []

    if len(g_lines) != len(a_lines):
        diffs.append(f"line count differs: golden={len(g_lines)} actual={len(a_lines)}")

    for i, (gl, al) in enumerate(zip(g_lines, a_lines, strict=False), start=1):
        g_tok = gl.split()
        a_tok = al.split()
        if len(g_tok) != len(a_tok):
            diffs.append(f"line {i}: token count differs\n  golden: {gl!r}\n  actual: {al!r}")
            continue
        for gt, at in zip(g_tok, a_tok, strict=True):
            gf, af = _try_float(gt), _try_float(at)
            if gf is not None and af is not None:
                if not math.isclose(gf, af, abs_tol=abs_tol, rel_tol=rel_tol):
                    diffs.append(
                        f"line {i}: numeric mismatch {gf} != {af} (abs_tol={abs_tol})"
                        f"\n  golden: {gl!r}\n  actual: {al!r}"
                    )
                    break
            elif gt != at:
                diffs.append(f"line {i}: text mismatch {gt!r} != {at!r}\n  golden: {gl!r}\n  actual: {al!r}")
                break

    return diffs
