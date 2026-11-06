"""Local Python headers for Triton gcc compiles and MLIP device helpers."""

from __future__ import annotations

import os
from pathlib import Path

from experiments.jacs_revision.lib.paths import repo_root


def mlip_device(use_gpu: bool) -> str:
    """Return ``cuda`` only when Figure 2 explicitly requests GPU."""
    return "cuda" if use_gpu else "cpu"


def _headers_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "tools" / "python312-headers"


def ensure_triton_python_headers(start: Path | None = None) -> dict:
    """Prepend a local ``Python.h`` directory to gcc include paths.

    Triton builds ``cuda_utils.c`` with ``-I/usr/include/python3.12`` on
    clusters that ship no dev headers for the venv Python.  We vendor the
    CPython 3.12.9 ``Include/`` tree under ``tools/python312-headers/``.
    """
    inc = _headers_dir(start)
    python_h = inc / "Python.h"
    if not python_h.is_file():
        return {"applied": False, "reason": f"missing {python_h}"}

    applied_paths = []
    prefix = str(inc)
    for key in ("CPATH", "C_INCLUDE_PATH"):
        prev = os.environ.get(key, "")
        parts = [p for p in prev.split(":") if p]
        if prefix not in parts:
            os.environ[key] = f"{prefix}:{prev}" if prev else prefix
            applied_paths.append(key)
    return {"applied": True, "include_dir": str(inc), "env_keys": applied_paths}


def make_aimnet2_calculator(device: str):
    """Build AIMNet2ASE on an explicit device without torch.compile."""
    try:
        from aimnet.calculators import AIMNet2ASE
        from aimnet.calculators.calculator import AIMNet2Calculator
    except ImportError:
        from aimnet.calculators.ase import AIMNet2ASE  # type: ignore[no-redef]
        from aimnet.calculators.calculator import AIMNet2Calculator

    base = AIMNet2Calculator("aimnet2", device=device, compile_model=False)
    return AIMNet2ASE(base)
