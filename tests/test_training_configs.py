"""Every config stack submit_water_train.sbatch can build must parse.

A training job is a 24 h GPU allocation, and a typo in a config file does not
show up until that allocation starts and run.py exits on it -- by which time the
slot is gone and the campaign has lost a day. The stacks are also layered
(base + theory overlay + cutoff overlay), so the failure mode is not just a
misspelled flag but an override landing in the wrong order and a run training on
the wrong labels or, worse, writing over another run's directory.

The parse is done in-process rather than by launching run.py so the test costs a
second and can assert on the resolved values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

BASE = "@config/training/water_wb97mv_001.txt"
PBE_OVERLAY = "@config/training/water_pbe_orca_overlay.txt"
MALON = "@config/training/malonaldehyde_pt_001.txt"
MALON_PBE_OVERLAY = "@config/training/malonaldehyde_pbe_orca_overlay.txt"

#: Mirrors the `case` blocks in submit_water_train.sbatch, including the
#: --args_file_name/--save_dir the script appends last.
STACKS: dict[str, list[str]] = {
    "water/wb97mv_def2tzvpd": [BASE],
    "water/pbe_d4_avdz": [BASE, PBE_OVERLAY],
    "malonaldehyde/wb97mv_def2tzvpd": [MALON],
    "malonaldehyde/pbe_d4_avdz": [MALON, MALON_PBE_OVERLAY],
    **{
        f"cutoff_{c}/pbe_d4_avdz": [BASE, PBE_OVERLAY, f"@config/training/water_clusters_cutoff_{c}.txt"]
        for c in (4, 5, 6, 8)
    },
}

RUN_NAMES = {
    "water/wb97mv_def2tzvpd": "water_wb97mv_def2tzvpd",
    "water/pbe_d4_avdz": "water_pbe_d4_avdz",
    "malonaldehyde/wb97mv_def2tzvpd": "malonaldehyde_wb97mv_def2tzvpd",
    "malonaldehyde/pbe_d4_avdz": "malonaldehyde_pbe_d4_avdz",
    **{f"cutoff_{c}/pbe_d4_avdz": f"water_pbe_orca_cutoff_{c}" for c in (4, 5, 6, 8)},
}


def parse_stack(stack: list[str], name: str | None = None):
    from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments

    argv = ["run.py", *stack]
    if name is not None:
        argv += [f"--args_file_name={name}", f"--save_dir=results/revision/{name}"]
    old = sys.argv
    old_cwd = Path.cwd()
    try:
        # The @-file paths in the stacks are repo-relative, as they are in the
        # submit script, which runs from SLURM_SUBMIT_DIR.
        import os

        os.chdir(_REPO_ROOT)
        sys.argv = argv
        args, _ = parse_command_line_arguments()
    finally:
        sys.argv = old
        import os

        os.chdir(old_cwd)
    return args


@pytest.mark.parametrize("key", sorted(STACKS))
def test_config_stack_parses(key):
    args = parse_stack(STACKS[key])
    assert args.max_steps > 0
    assert args.cutoff > 0


@pytest.mark.parametrize("key", sorted(STACKS))
def test_stack_references_files_that_exist(key):
    """The prior and basis tables must be on disk now; labels may still be running.

    Splitting the check this way is the point: a missing SAD prior or orbital
    table is a mistake in the config that will never fix itself, whereas a
    missing label file just means the campaign has not got there yet.
    """
    args = parse_stack(STACKS[key])
    for attr in ("atom_dens_path", "orbitals_file", "radial_coeffs_file"):
        path = _REPO_ROOT / getattr(args, attr)
        assert path.is_file(), f"{key}: {attr} missing at {path}"


@pytest.mark.parametrize("key", sorted(STACKS))
def test_each_run_writes_to_its_own_directory(key):
    """Two runs sharing a save_dir would overwrite each other's checkpoints.

    The cutoff sweep is where this bites: all four variants inherit the same
    --save_dir from the PBE overlay and are only separated by the override the
    submit script appends, so this asserts that override actually wins.
    """
    name = RUN_NAMES[key]
    args = parse_stack(STACKS[key], name=name)
    assert args.save_dir == f"results/revision/{name}"
    assert args.args_file_name == name


def test_cutoff_overlay_changes_only_the_cutoff():
    """The sweep answers 'how far must message passing reach', so nothing else may move."""
    base = parse_stack(STACKS["water/pbe_d4_avdz"])
    for cutoff in (4, 5, 6, 8):
        swept = parse_stack(STACKS[f"cutoff_{cutoff}/pbe_d4_avdz"])
        assert swept.cutoff == float(cutoff)
        differing = {
            k
            for k, v in vars(swept).items()
            if vars(base).get(k) != v and k not in {"cutoff", "save_dir", "args_file_name"}
        }
        assert not differing, f"cutoff {cutoff} also changed {sorted(differing)}"


def test_the_two_theories_differ_only_in_data_and_destination():
    """The PBE-vs-hybrid comparison only isolates the reference data if the model does not move.

    This is the claim the overlay exists to support: same architecture, same
    optimiser, different labels. Anything else that differs would let a reviewer
    attribute the gap to the network rather than the functional.
    """
    hybrid = parse_stack(STACKS["water/wb97mv_def2tzvpd"])
    pbe = parse_stack(STACKS["water/pbe_d4_avdz"])
    allowed = {
        "np_dataset",
        "dens_dataset",
        "np_dataset_valid",
        "dens_dataset_valid",
        "np_dataset_test",
        "dens_dataset_test",
        "atom_dens_path",
        "save_dir",
        "args_file_name",
    }
    differing = {k for k, v in vars(pbe).items() if vars(hybrid).get(k) != v} - allowed
    assert not differing, f"PBE overlay also changed {sorted(differing)}"


def test_label_paths_carry_the_theory_they_were_computed_at():
    """Training on the other functional's labels would be silent and invalidate the comparison."""
    for key, theory in (
        ("water/wb97mv_def2tzvpd", "wb97mv_def2tzvpd"),
        ("water/pbe_d4_avdz", "pbe_d4_avdz"),
        ("malonaldehyde/wb97mv_def2tzvpd", "wb97mv_def2tzvpd"),
        ("malonaldehyde/pbe_d4_avdz", "pbe_d4_avdz"),
    ):
        args = parse_stack(STACKS[key])
        for attr in ("dens_dataset", "dens_dataset_valid", "dens_dataset_test"):
            assert theory in getattr(args, attr), f"{key}: {attr} is not a {theory} label set"
