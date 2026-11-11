"""Matplotlib classic + Computer Modern for response-letter figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

_RESPONSE_FIG_EXT = ".pdf"

_RC = {
    "text.usetex": True,
    "font.family": "serif",
    # cmr10 is TeX-encoded and drops spaces/underscores. Use a Unicode serif
    # for words; mathtext CM still draws $...$ in Computer Modern.
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Computer Modern Roman", "cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "legend.frameon": False,
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def response_figure_path(path: str | Path) -> Path:
    """Normalize a response-letter figure path to vector PDF."""
    dest = Path(path)
    if dest.suffix.lower() == ".png":
        dest = dest.with_suffix(_RESPONSE_FIG_EXT)
    elif dest.suffix == "":
        dest = dest.with_suffix(_RESPONSE_FIG_EXT)
    return dest


def save_response_figure(fig, path: str | Path, **kwargs) -> Path:
    """Save a matplotlib figure as vector PDF for LaTeX inclusion."""
    dest = response_figure_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"format": "pdf", "bbox_inches": "tight", "facecolor": "white"}
    save_kwargs.update(kwargs)
    fig.savefig(dest, **save_kwargs)
    plt.close(fig)
    return dest


def usetex_available() -> bool:
    """Return True when a traditional LaTeX + dvipng toolchain is on PATH."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def apply() -> None:
    """Use classic ticks and Computer Modern text on a white face.

    Existing series colors stay hardcoded at the call site. If ``usetex``
    cannot run, fall back to mathtext CM so labels do not mix Arial and TeX.
    """
    if mpl.get_backend().lower() != "agg":
        mpl.use("Agg")
    plt.style.use("classic")
    rc = dict(_RC)
    if not usetex_available():
        rc["text.usetex"] = False
    else:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    mpl.rcParams.update(rc)
