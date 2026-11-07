#!/usr/bin/env python3
"""Compatibility entry point; implementation: experiments.jacs_revision.analysis.run_mace_ethanol_ir."""
from pathlib import Path
import sys

for _root in Path(__file__).absolute().parents:
    if (_root / "pyproject.toml").is_file():
        sys.path.insert(0, str(_root))
        sys.path.insert(0, str(_root / "src"))
        break

if __name__ == "__main__":
    from equiv_dens.entrypoints import run_module
    run_module("experiments.jacs_revision.analysis.run_mace_ethanol_ir")
else:
    from equiv_dens.entrypoints import forward_module
    forward_module(globals(), "experiments.jacs_revision.analysis.run_mace_ethanol_ir")
