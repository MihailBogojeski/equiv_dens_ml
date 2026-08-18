#!/usr/bin/env python3
"""Extract official QM7-X geometries and bookkeeping properties from HDF5.

Writes one JSONL row per structure (id, Z, R, official ePBE0+MBD / totFOR).
DensNet energy / force / density labels still come from the ORCA worker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import iter_qm7x_records, write_jsonl  # noqa: E402


def discover_hdf5(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.hdf5")) + sorted(raw_dir.glob("*.h5"))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=str(_REPO_ROOT / "datasets/revision/qm7x/raw"),
        help="Directory of extracted QM7-X HDF5 files",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "datasets/revision/qm7x/qm7x_official.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--max-records", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = discover_hdf5(raw_dir)
    if not files:
        raise SystemExit(f"no HDF5 files in {raw_dir}; run scripts/revision/download_qm7x.sh")

    rows = []
    for path in files:
        for rec in iter_qm7x_records(path):
            rows.append(rec)
            if args.max_records and len(rows) >= args.max_records:
                break
        if args.max_records and len(rows) >= args.max_records:
            break

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} records to {args.out} from {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
