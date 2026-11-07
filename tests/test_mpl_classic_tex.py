"""Response-letter matplotlib export helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

from mpl_classic_tex import response_figure_path, save_response_figure  # noqa: E402


def test_response_figure_path_normalizes_png_to_pdf():
    assert response_figure_path("foo/bar.png") == Path("foo/bar.pdf")
    assert response_figure_path("foo/bar.pdf") == Path("foo/bar.pdf")


def test_save_response_figure_writes_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    dest = save_response_figure(fig, tmp_path / "example.png")
    assert dest == tmp_path / "example.pdf"
    assert dest.is_file()
    assert dest.read_bytes()[:4] == b"%PDF"
