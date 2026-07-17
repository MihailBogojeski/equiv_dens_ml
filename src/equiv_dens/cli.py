"""CLI for run.py and the equiv-dens entry point.

Subcommands dispatch to scripts/training/ and scripts/md/. Run with
``python run.py --help`` or ``equiv-dens --help`` after pip install.
"""

import equiv_dens.compat  # noqa: F401 - apply patches before any schnetpack import

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = str(_REPO_ROOT / "src")


def _env_with_src():
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


TRAIN_SCRIPTS = {
    "density": "scripts/training/train_dens.py",
    "joint": "scripts/training/train.py",
    "multiphase": "scripts/training/train_all.py",
    "energy": "scripts/training/train_only_en.py",
}


def _cmd_train(args: argparse.Namespace) -> int:
    config = args.config
    extra = args.extra or []
    if not config and extra:
        config, extra = extra[0], extra[1:]
    if not config:
        print(
            "Error: provide config, e.g. equiv-dens train @config/training/h2o_small_all_001.txt",
            file=sys.stderr,
        )
        return 1
    config_arg = config if config.startswith("@") else f"@{config}"
    script = _REPO_ROOT / TRAIN_SCRIPTS[args.mode]
    cmd = [sys.executable, str(script), config_arg] + extra
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def _cmd_md(args: argparse.Namespace) -> int:
    config = args.config
    extra = args.extra or []
    if not config and extra:
        config, extra = extra[0], extra[1:]
    if not config:
        print(
            "Error: provide config, e.g. equiv-dens md @config/md/nn/polythiophene_2mer.txt",
            file=sys.stderr,
        )
        return 1
    config_arg = config if config.startswith("@") else f"@{config}"
    script = _REPO_ROOT / "scripts" / "md" / "schnetpack_md_run.py"
    cmd = [sys.executable, str(script), config_arg] + extra
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def _cmd_gxtb_md(args: argparse.Namespace) -> int:
    config = args.config
    extra = args.extra or []
    if not config and extra:
        config, extra = extra[0], extra[1:]
    if not config:
        print(
            "Error: provide config, e.g. equiv-dens gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt",
            file=sys.stderr,
        )
        return 1
    config_arg = config if config.startswith("@") else f"@{config}"
    script = _REPO_ROOT / "scripts" / "md" / "gxtb_md_run.py"
    cmd = [sys.executable, str(script), config_arg] + extra
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def _cmd_infer(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    args_file = model_path / "args.txt" if model_path.is_dir() else model_path
    if not args_file.exists():
        print(f"Error: args file not found: {args_file}", file=sys.stderr)
        return 1
    traj = Path(args.trajectory)
    if not traj.exists():
        print(f"Error: trajectory not found: {traj}", file=sys.stderr)
        return 1
    script = _REPO_ROOT / "scripts" / "training" / "eval_model_npy.py"
    cmd = [
        sys.executable,
        str(script),
        str(args_file),
        str(traj),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.dpm_intor:
        cmd.append("--dpm_intor")
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def _cmd_test(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    args_file = checkpoint / "args.txt" if checkpoint.is_dir() else checkpoint
    if not args_file.exists():
        print(f"Error: args file not found: {args_file}", file=sys.stderr)
        return 1
    script = _REPO_ROOT / "tests" / "eval_model.py"
    cmd = [sys.executable, str(script), f"@{args_file}"] + (args.extra or [])
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def _cmd_dipole(args: argparse.Namespace) -> int:
    script = _REPO_ROOT / "scripts" / "recompute_polythiophene_dipoles_parallel.py"
    if not script.exists():
        print(f"Error: {script} not found", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(script),
        "--trajectory",
        str(args.trajectory),
        "--model_path",
        str(args.model),
        "--oligomer",
        args.oligomer or "8mer",
        "--replica",
        str(args.replica or 0),
        "--start_frame",
        str(args.start_frame or 0),
        "--end_frame",
        str(args.end_frame or 10),
        "--output_dir",
        str(args.output_dir or "scratch/dipole"),
    ]
    return subprocess.call(cmd, cwd=str(_REPO_ROOT), env=_env_with_src())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="equiv_dens_ml unified entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unified entry point for equiv_dens_ml.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument("--mode", choices=TRAIN_SCRIPTS, default="joint")
    p_train.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Args config (e.g. @config/training/h2o_small_all_001.txt)",
    )
    p_train.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to train script")
    p_train.set_defaults(func=_cmd_train)

    p_md = subparsers.add_parser("md", help="Run ML-enhanced molecular dynamics")
    p_md.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Args config (e.g. @config/md/nn/polythiophene_2mer.txt)",
    )
    p_md.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to schnetpack_md_run")
    p_md.set_defaults(func=_cmd_md)

    p_gxtb_md = subparsers.add_parser("gxtb-md", help="Run g-xtb MD (no model)")
    p_gxtb_md.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Args config (e.g. @config/md/gxtb/gxtb_polythiophene_2mer.txt)",
    )
    p_gxtb_md.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to gxtb_md_run")
    p_gxtb_md.set_defaults(func=_cmd_gxtb_md)

    p_infer = subparsers.add_parser("infer", help="Inference on trajectory (density, energy, forces, dipole)")
    p_infer.add_argument("--model", "-m", required=True, help="Model dir (args.txt) or args file")
    p_infer.add_argument("--trajectory", "-t", required=True, help="Trajectory .npy")
    p_infer.add_argument("--batch-size", type=int, default=1)
    p_infer.add_argument("--dpm-intor", action="store_true")
    p_infer.set_defaults(func=_cmd_infer)

    p_test = subparsers.add_parser("test", help="Evaluate on dataset")
    p_test.add_argument(
        "--checkpoint",
        "-C",
        type=str,
        required=True,
        help="Path to model directory (contains args.txt)",
    )
    p_test.add_argument("extra", nargs="*", help="Extra args passed to test.py")
    p_test.set_defaults(func=_cmd_test)

    p_dipole = subparsers.add_parser("dipole", help="Dipole recomputation (parallel/SLURM)")
    p_dipole.add_argument("--trajectory", "-t", type=str, required=True, help="Path to trajectory .npy or .xyz")
    p_dipole.add_argument("--model", "-m", type=str, required=True, help="Path to polythiophene model")
    p_dipole.add_argument("--oligomer", type=str, default="8mer", help="Oligomer type (8mer, 10mer, 12mer)")
    p_dipole.add_argument("--replica", type=int, default=0, help="Replica index")
    p_dipole.add_argument("--start_frame", type=int, default=0, help="Start frame (inclusive)")
    p_dipole.add_argument("--end_frame", type=int, default=10, help="End frame (exclusive)")
    p_dipole.add_argument("--output_dir", "-o", type=str, default="scratch/dipole", help="Output directory")
    p_dipole.set_defaults(func=_cmd_dipole)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
