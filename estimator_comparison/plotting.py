"""Deterministic Matplotlib output (PDF + PNG) with a fixed, validated palette."""
from __future__ import annotations

from pathlib import Path

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"within_margin": "#0ca30c", "above_margin": "#fab219", "memory_failure": "#d03b3b",
          "unresolved": "#898781"}
STATE_MARKERS = {"within_margin": "o", "above_margin": "^", "memory_failure": "X", "unresolved": "s"}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def setup():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.hashsalt": "rlvrambench-review",
    })
    import matplotlib.pyplot as plt

    return plt


def save(fig, directory: Path, stem: str) -> list[str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{stem}.pdf"
    png = directory / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None,
                                                    "Producer": None, "Creator": None})
    fig.savefig(png, dpi=200, bbox_inches="tight", metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return [pdf.name, png.name]
