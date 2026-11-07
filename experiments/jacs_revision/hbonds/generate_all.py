#!/usr/bin/env python3
"""Generate all H-bond / new-system XYZ trees and run overlap gates."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_all.py'


import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402

_REPO_ROOT = repo_root()

SYSTEMS = [
    ("generate_meoh_water.py", "meoh_water", "datasets/revision/meoh_water"),
    ("generate_acac.py", "acac", "datasets/revision/acac"),
    ("generate_paracetamol.py", "paracetamol", "datasets/revision/paracetamol"),
    ("generate_nma_dimer.py", "nma_dimer", "datasets/revision/nma_dimer"),
    ("generate_pgg.py", "pgg", "datasets/revision/pgg"),
    ("generate_crambin_frags.py", "crambin_frags", "datasets/revision/crambin_frags"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="pass --quick to each generator")
    args = parser.parse_args()

    py = sys.executable
    failed = False
    for script, system, rel_dir in SYSTEMS:
        out_dir = _REPO_ROOT / rel_dir
        if not args.gates_only:
            print(f"=== generating {system} ===")
            rc = subprocess.call([py, str(_SCRIPT_DIR / script), *(["--quick"] if args.quick else [])], cwd=_REPO_ROOT)
            if rc != 0:
                print(f"generator failed for {system}", file=sys.stderr)
                failed = True
                continue
        print(f"=== gates {system} ===")
        rc = subprocess.call(
            [py, str(_SCRIPT_DIR / "overlap_gates.py"), "--system-dir", str(out_dir), "--system", system],
            cwd=_REPO_ROOT,
        )
        if rc != 0:
            print(f"gates failed for {system}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
