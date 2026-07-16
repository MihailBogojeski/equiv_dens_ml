#!/usr/bin/env python3
"""Unified entry point for equiv_dens_ml. Delegates to equiv_dens.cli."""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = str(_REPO_ROOT / "src")

# Ensure equiv_dens is importable when run from repo root
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from equiv_dens.cli import main

if __name__ == "__main__":
    sys.exit(main())
