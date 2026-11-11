#!/usr/bin/env python3
"""Compatibility entry point; implementation: equiv_dens.commands.gxtb_md."""
from pathlib import Path
import sys

for _root in Path(__file__).absolute().parents:
    if (_root / "pyproject.toml").is_file():
        sys.path.insert(0, str(_root / "src"))
        break

if __name__ == "__main__":
    from equiv_dens.commands.gxtb_md import main
    raise SystemExit(main())
else:
    # Preserve helper imports, including private names and module identity.
    from importlib import import_module
    _implementation = import_module("equiv_dens.commands.gxtb_md")
    sys.modules[__name__] = _implementation
    globals().update(vars(_implementation))
