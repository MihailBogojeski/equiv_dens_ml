#!/usr/bin/env python3
"""Add PySCF mo_basis entries so schnetpack MD can join SAD MO coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pyscf import gto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sad", nargs="+", type=Path)
    parser.add_argument("--basis", default="augccpvdz")
    args = parser.parse_args()

    for path in args.sad:
        sad = np.load(path, allow_pickle=True).item()
        for z, rec in sad.items():
            if not isinstance(rec, dict):
                continue
            if "mo_basis" in rec and int(z) in rec["mo_basis"]:
                continue
            symbol = rec.get("symbol") or gto.charge(int(z))
            if isinstance(symbol, int):
                symbol = gto.mole.ELEMENTS[int(z)]
            mol = gto.M(atom=[[int(z), [0, 0, 0]]], basis=args.basis, spin=int(z) % 2)
            rec["mo_basis"] = {int(z): mol._basis[symbol]}
        np.save(path, sad, allow_pickle=True)
        print(f"patched {path} zs={list(sad)}")


if __name__ == "__main__":
    main()
