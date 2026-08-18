#!/usr/bin/env python3
"""Build the stored->PySCF AO permutation from ORCA's printed AO labels.

ORCA writes its basis functions in a different order from PySCF in two ways:

  1. the def2-TZVPD diffuse augmentation is appended at the end of each atom's
     block rather than merged into the matching angular-momentum group, so e.g.
     carbon runs 1s..5s, p, d, f, *then* 6s and the diffuse d shell;
  2. within a shell the components run m = 0, +1, -1, +2, -2, ... instead of
     PySCF's m-ascending (and PySCF's (x, y, z) for l=1).

Both are read straight off the ``MOLECULAR ORBITALS`` block of an ORCA output
rather than assumed, so this does not depend on guessing the convention.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

_AO_RE = re.compile(r"^\s{2}(\d+)([A-Za-z]{1,2})\s+(\S+)\s")

# ORCA component name -> magnetic quantum number, per angular momentum.
_COMPONENT_M = {
    0: {"s": 0},
    1: {"pz": 0, "px": 1, "py": -1},
    2: {"dz2": 0, "dxz": 1, "dyz": -1, "dx2y2": 2, "dxy": -2},
}
_L_OF_LETTER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}


def parse_orca_ao_labels(path: str | Path) -> list[tuple[int, int, int, int]]:
    """Return, in ORCA order, one (atom, l, shell_index, m) tuple per AO."""
    labels: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    started = False
    for line in Path(path).read_text().splitlines():
        if "MOLECULAR ORBITALS" in line:
            started = True
            continue
        if not started:
            continue
        match = _AO_RE.match(line)
        if not match:
            continue
        key = (int(match.group(1)), match.group(2), match.group(3))
        if key in seen:
            continue
        seen.add(key)
        labels.append(key)

    parsed = []
    for atom, _element, orbital in labels:
        shell = int(re.match(r"\d+", orbital).group(0))
        rest = orbital[len(str(shell)) :]
        l = _L_OF_LETTER[rest[0]]
        if l <= 2:
            m = _COMPONENT_M[l][rest]
        else:
            m = int(rest[1:])
        parsed.append((atom, l, shell, m))
    return parsed


def pyscf_ao_index(mol) -> dict[tuple[int, int, int, int], int]:
    """Map (atom, l, shell_index, m) -> PySCF AO index. Shell index is 1-based per l."""
    ao_loc = mol.ao_loc_nr()
    counters: dict[tuple[int, int], int] = {}
    index = {}
    for shell in range(mol.nbas):
        atom = mol.bas_atom(shell)
        l = mol.bas_angular(shell)
        counters[(atom, l)] = counters.get((atom, l), 0) + 1
        n = counters[(atom, l)]
        ms = [1, -1, 0] if l == 1 else list(range(-l, l + 1))
        for j, m in enumerate(ms):
            index[(atom, l, n, m)] = ao_loc[shell] + j
    return index


def permutation_from_orca(mol, orca_out: str | Path) -> np.ndarray:
    """perm[i] = stored (ORCA) index of the AO PySCF places at index i."""
    parsed = parse_orca_ao_labels(orca_out)
    if len(parsed) != mol.nao:
        raise ValueError(f"ORCA lists {len(parsed)} AOs, PySCF has {mol.nao}")
    lookup = pyscf_ao_index(mol)
    perm = np.empty(mol.nao, dtype=int)
    for orca_index, key in enumerate(parsed):
        perm[lookup[key]] = orca_index
    if sorted(perm.tolist()) != list(range(mol.nao)):
        raise ValueError("recovered map is not a permutation")
    return perm
