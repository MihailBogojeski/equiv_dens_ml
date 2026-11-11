"""Discover this installation's checkout without assuming directory depth."""
from pathlib import Path


def checkout_root(start=None):
    """Return the source checkout, or None for a standalone installation."""
    here = Path(start or __file__).resolve()
    for candidate in (here.parent, *here.parents):
        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "src/equiv_dens/cli.py").is_file()):
            return candidate
    return None


def project_root():
    """Use this source checkout, otherwise the caller's working directory."""
    return checkout_root() or Path.cwd()
