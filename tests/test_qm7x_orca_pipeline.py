"""Unit tests for the QM7-X ORCA high-throughput pipeline (no live ORCA)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "revision"
sys.path.insert(0, str(SCRIPT_DIR))

from qm7x_assemble_npy import load_shard_results  # noqa: E402
from qm7x_build_shards import annotate_shell, frames_from_npy, shard_frames, slurm_array_ranges  # noqa: E402
from qm7x_orca_common import (  # noqa: E402
    THEORY_KEYWORDS,
    XC,
    build_pyscf_mol,
    calc_dict_from_orca,
    electron_count,
    forces_from_gradient,
    is_closed_shell,
    iter_qm7x_records,
    pad_frames,
    parse_json_labels,
    parse_orca_energy,
    parse_orca_engrad,
    parse_orca_gradient,
    pyscf_ao_index,
    unpadded_atoms,
    write_orca_inp,
    write_shard,
)
from qm7x_orca_worker import process_frame  # noqa: E402
from theory_levels import DEFAULT_LEVEL, get_level, level_keys  # noqa: E402

SAMPLE_OUT = """\
                              * O   R   C   A *
some preamble
FINAL SINGLE POINT ENERGY      -76.111111111111
CARTESIAN GRADIENT
------------------

   1   O   :    0.010000000   -0.020000000    0.030000000
   2   H   :   -0.004000000    0.005000000   -0.006000000
   3   H   :   -0.006000000    0.015000000   -0.024000000

Norm of the cartesian gradient ...
FINAL SINGLE POINT ENERGY      -76.424242424242
"""

WATER_Z = [8, 1, 1]
WATER_XYZ = [
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
]


def test_unpadded_atoms_drops_zeros():
    z = [6, 1, 1, 0, 0]
    xyz = np.arange(15, dtype=float).reshape(5, 3)
    z2, xyz2 = unpadded_atoms(z, xyz)
    assert z2.tolist() == [6, 1, 1]
    assert xyz2.shape == (3, 3)


def test_closed_shell_water_and_open_shell_ch():
    assert is_closed_shell(WATER_Z)
    assert electron_count(WATER_Z) == 10
    assert not is_closed_shell([6, 1])
    assert electron_count([6, 1]) == 7


def test_write_orca_inp_pbe0_augccpvdz(tmp_path):
    path = write_orca_inp(tmp_path / "job.inp", WATER_Z, WATER_XYZ, nprocs=8, maxcore_mb=4000)
    text = path.read_text()
    assert f"! {THEORY_KEYWORDS}" in text
    assert "%maxcore 4000" in text
    assert "nprocs 8" in text
    assert "* xyz 0 1" in text
    assert "  8 " in text
    with pytest.raises(ValueError, match="open-shell"):
        write_orca_inp(tmp_path / "bad.inp", [6, 1], [[0, 0, 0], [1, 0, 0]])


def test_parse_orca_energy_uses_last_value():
    assert parse_orca_energy(SAMPLE_OUT) == pytest.approx(-76.424242424242)


def test_parse_orca_gradient_and_forces():
    grad = parse_orca_gradient(SAMPLE_OUT)
    assert grad.shape == (3, 3)
    assert grad[0, 0] == pytest.approx(0.01)
    forces = forces_from_gradient(grad)
    import ase

    assert forces == pytest.approx(-grad / ase.units.Bohr)


def test_extract_official_hdf5(tmp_path):
    h5_path = tmp_path / "8000.hdf5"
    with h5py.File(h5_path, "w") as handle:
        conf = handle.create_group("1").create_group("Geom-m1-i1-c1-opt")
        conf.create_dataset("atNUM", data=np.array(WATER_Z, dtype=int))
        conf.create_dataset("atXYZ", data=np.array(WATER_XYZ, dtype=float))
        conf.create_dataset("ePBE0+MBD", data=np.array([-2083.0]))
        conf.create_dataset("totFOR", data=np.ones((3, 3)))
    recs = list(iter_qm7x_records(h5_path))
    assert len(recs) == 1
    assert recs[0]["qm7x_id"] == "1/Geom-m1-i1-c1-opt"
    assert recs[0]["atom_numbers"] == WATER_Z
    assert recs[0]["ePBE0_MBD_eV"] == pytest.approx(-2083.0)
    assert np.asarray(recs[0]["totFOR_eV_A"]).shape == (3, 3)


def test_build_shards_from_npy_skips_open_shell(tmp_path):
    n = 5
    positions = np.zeros((n, 4, 3))
    atom_numbers = np.zeros((n, 4), dtype=int)
    for i in range(4):
        atom_numbers[i, :3] = WATER_Z
        positions[i, :3] = WATER_XYZ
    atom_numbers[4, :2] = [6, 1]
    positions[4, :2] = [[0, 0, 0], [1.1, 0, 0]]
    npy = tmp_path / "base.npy"
    np.save(npy, {"positions": positions, "atom_numbers": atom_numbers, "atom_types": [1, 6, 8]})
    frames = frames_from_npy(npy, max_frames=0, offset=0)
    kept, skipped = annotate_shell(frames)
    assert len(kept) == 4
    assert len(skipped) == 1
    shards = shard_frames(kept, 2)
    assert [len(s) for s in shards] == [2, 2]
    assert [len(s) for s in shard_frames(kept, 1)] == [1, 1, 1, 1]
    assert slurm_array_ranges(100) == [(0, 99)]
    assert slurm_array_ranges(20000) == [(0, 9999), (10000, 19999)]
    assert slurm_array_ranges(20000, 2500) == [
        (0, 2499),
        (2500, 4999),
        (5000, 7499),
        (7500, 9999),
        (10000, 12499),
        (12500, 14999),
        (15000, 17499),
        (17500, 19999),
    ]
    dest = tmp_path / "shard_0000.json"
    write_shard(dest, 0, "smoke", shards[0])
    loaded = json.loads(dest.read_text())
    assert loaded["split"] == "smoke"
    assert loaded["frames"][0]["closed_shell"] is True


def test_pad_and_assemble_roundtrip(tmp_path):
    frames = [
        {
            "atom_numbers": WATER_Z,
            "positions": WATER_XYZ,
            "energy": -76.4,
            "forces": np.zeros((3, 3)),
            "dipole": [0.0, 0.0, 0.1],
        }
    ]
    packed = pad_frames(frames)
    assert packed["positions"].shape == (1, 3, 3)
    assert packed["energy"][0] == pytest.approx(-76.4)
    shard = tmp_path / "shard_0000"
    shard.mkdir()
    mol = build_pyscf_mol(WATER_Z, WATER_XYZ)
    calc = {"energy": -76.4, "forces": np.zeros((3, 3)), "dipole": np.array([0.0, 0.0, 0.1])}
    np.save(shard / "results.npy", np.array([{"index": 0, "mol": mol.pack(), "calc": calc}], dtype=object))
    loaded = load_shard_results(shard)
    assert loaded[0]["index"] == 0
    assert loaded[0]["calc"]["energy"] == pytest.approx(-76.4)


def test_calc_dict_from_orca_without_df():
    mol = build_pyscf_mol(WATER_Z, WATER_XYZ)
    nao = mol.nao
    nocc = mol.nelectron // 2
    mo_coeff = np.eye(nao)
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2.0
    calc = calc_dict_from_orca(
        mol,
        -76.4,
        np.zeros((3, 3)),
        mo_coeff,
        mo_occ,
        fit_df=False,
    )
    assert calc["xc"] == XC
    assert calc["energy"] == pytest.approx(-76.4)
    assert calc["mo_coeff"].shape == (nao, nao)
    assert calc["dipole"].shape == (3,)
    assert "df_coeff" not in calc


def test_parse_orca_engrad():
    text = """\
#
# Number of atoms
#
 2
#
# The current total energy in Eh
#
   -1.125000000000
#
# The current gradient in Eh/bohr
#
      0.010000000000
      0.000000000000
      0.000000000000
     -0.010000000000
      0.000000000000
      0.000000000000
#
# The atomic numbers and current coordinates in Bohr
#
   1     0.0 0.0 0.0
   1     1.4 0.0 0.0
"""
    energy, grad = parse_orca_engrad(text)
    assert energy == pytest.approx(-1.125)
    assert grad.shape == (2, 3)
    assert grad[0, 0] == pytest.approx(0.01)


def test_pyscf_ao_index_counts_general_contraction():
    mol = build_pyscf_mol([6], [[0.0, 0.0, 0.0]])
    lookup = pyscf_ao_index(mol)
    s_shells = {key[2] for key in lookup if key[0] == 0 and key[1] == 0}
    assert s_shells == {1, 2, 3, 4}
    assert len(lookup) == mol.nao


def test_parse_json_labels_matches_orca_print():
    labels = ["0O   1s", "0O   1pz", "0O   1px", "0O   1py", "1H   1s"]
    parsed = parse_json_labels(labels)
    assert parsed[0] == (0, 0, 1, 0)
    assert parsed[1] == (0, 1, 1, 0)
    assert parsed[2] == (0, 1, 1, 1)
    assert parsed[3] == (0, 1, 1, -1)
    assert parsed[4] == (1, 0, 1, 0)


def test_worker_dry_run_and_open_shell_skip(tmp_path):
    level = get_level(DEFAULT_LEVEL)
    rec = process_frame(
        {"index": 0, "atom_numbers": WATER_Z, "positions": WATER_XYZ},
        tmp_path / "ok",
        level=level,
        orca_bin="orca",
        orca2json="orca_2json",
        nprocs=8,
        maxcore_mb=4000,
        fit_df=False,
        dry_run=True,
        keep_orca=True,
    )
    assert rec["status"] == "dry_run"
    assert (tmp_path / "ok" / "job.inp").is_file()
    assert level.orca_keywords in (tmp_path / "ok" / "job.inp").read_text()

    skipped = process_frame(
        {"index": 1, "atom_numbers": [6, 1], "positions": [[0, 0, 0], [1.1, 0, 0]]},
        tmp_path / "skip",
        level=level,
        orca_bin="orca",
        orca2json="orca_2json",
        nprocs=8,
        maxcore_mb=4000,
        fit_df=False,
        dry_run=True,
        keep_orca=True,
    )
    assert skipped["status"] == "skipped_open_shell"


def test_every_registered_theory_writes_an_orca_input(tmp_path):
    """The registry is what a --theory flag selects, so each entry must be usable.

    The dry run stops short of calling ORCA, but it exercises the part that a
    typo in the registry would break: the keyword line reaching the input file.
    """
    for key in level_keys():
        level = get_level(key)
        if "orca" not in level.engines:
            continue
        rec = process_frame(
            {"index": 0, "atom_numbers": WATER_Z, "positions": WATER_XYZ},
            tmp_path / key,
            level=level,
            orca_bin="orca",
            orca2json="orca_2json",
            nprocs=8,
            maxcore_mb=4000,
            fit_df=False,
            dry_run=True,
            keep_orca=True,
        )
        assert rec["status"] == "dry_run"
        assert rec["theory"] == key
        assert level.orca_keywords in (tmp_path / key / "job.inp").read_text()
