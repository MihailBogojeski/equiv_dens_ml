"""Locate the repository and the revision ``lib`` directory.

Moved campaign scripts no longer share a single ``parent.parent`` depth, so
walk up to ``pyproject.toml`` instead of counting directories.
"""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the checkout root that contains ``pyproject.toml``."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"could not find repo root from {here}")


def revision_lib(start: Path | None = None) -> Path:
    """Return ``scripts/revision/lib``."""
    return repo_root(start) / "scripts" / "revision" / "lib"


def ensure_revision_imports(*extra: Path) -> None:
    """Put ``lib`` (and optional same-theme dirs) on ``sys.path``."""
    import sys

    for path in (revision_lib(), *extra):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
