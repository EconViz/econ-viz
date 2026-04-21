"""Unicode and LaTeX-escape handling for text nodes in TikZ output.

pdfLaTeX rejects several Unicode codepoints that Matplotlib emits by
default — most notably U+2212 (minus sign) used in negative tick labels.
Every string that flows through ``TikzRenderer.draw_text`` is normalised
here before being embedded in a ``\\node`` command.
"""

from __future__ import annotations

_UNICODE_FIXES = str.maketrans(
    {
        "\u2212": "-",     # MINUS SIGN
        "\u2013": "--",    # EN DASH
        "\u2014": "---",   # EM DASH
        "\u00a0": "~",     # NO-BREAK SPACE
        "\u2018": "`",     # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",     # RIGHT SINGLE QUOTATION MARK
        "\u201c": "``",    # LEFT DOUBLE QUOTATION MARK
        "\u201d": "''",    # RIGHT DOUBLE QUOTATION MARK
        "\u2032": "'",     # PRIME
        "\u2033": "''",    # DOUBLE PRIME
    }
)

_LATEX_SPECIALS = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
)


def sanitize_text(s: str, *, ismath: bool) -> str:
    """Return a pdfLaTeX-safe version of ``s``.

    For math-mode text (``ismath=True``) only Unicode normalisation is
    applied, so legitimate TeX commands and braces pass through untouched.
    For plain text, LaTeX-special characters are additionally escaped.
    """
    out = s.translate(_UNICODE_FIXES)
    if not ismath:
        out = out.translate(_LATEX_SPECIALS)
    return out
