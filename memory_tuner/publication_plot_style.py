"""Shared Matplotlib settings for publisher-ready vector figures."""

from __future__ import annotations


def configure_matplotlib() -> None:
    """Embed TrueType fonts rather than Matplotlib's default Type-3 glyphs."""

    import matplotlib

    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
