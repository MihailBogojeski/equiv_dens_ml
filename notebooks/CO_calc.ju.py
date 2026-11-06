#!/usr/bin/env python3
"""Compatibility entry point for historical research code."""
from pathlib import Path
import sys

_legacy_file = Path(__file__).absolute()
for _root in _legacy_file.parents:
    if (_root / "pyproject.toml").is_file():
        break
_source = _root / 'experiments/legacy/notebooks/CO_calc.ju.py'
# Keep the old logical filename, import location, argv and working directory.
exec(compile(_source.read_bytes(), str(_legacy_file), "exec"), globals(), globals())
