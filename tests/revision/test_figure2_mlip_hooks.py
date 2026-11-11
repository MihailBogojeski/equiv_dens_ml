"""Tests for the Reviewer 3 MLIP timing additions."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_ANALYSIS = _REPO / "scripts" / "revision" / "analysis"
sys.path.insert(0, str(_ANALYSIS))


def test_figure2_benchmark_recognizes_requested_mlips(implementation_text) -> None:
    text = implementation_text(_ANALYSIS / "benchmark_figure2.py")
    assert 'method == "so3lr"' in text
    assert 'method == "macepolar"' in text
    assert "mace_polar" in text
    assert 'model="polar-1-m"' in text


def test_gpu_bar_requires_clear_speedup() -> None:
    from timing_policy import gpu_speedup_is_clear

    assert gpu_speedup_is_clear(cpu_mean_s=1.0, gpu_mean_s=0.79)
    assert not gpu_speedup_is_clear(cpu_mean_s=1.0, gpu_mean_s=0.80)
    assert not gpu_speedup_is_clear(cpu_mean_s=1.0, gpu_mean_s=1.1)
    assert not gpu_speedup_is_clear(cpu_mean_s=0.0, gpu_mean_s=0.1)
