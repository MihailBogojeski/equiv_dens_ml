"""Run explicitly registered, checkout-local experiment tasks."""

import json
import os
from pathlib import Path
import subprocess
import sys

from equiv_dens.paths import checkout_root


def load_registry(root):
    campaign = Path(root) / "experiments" / "jacs_revision"
    return campaign, json.loads((campaign / "tasks.json").read_text())


def dispatch(args):
    root = checkout_root() or checkout_root(Path.cwd() / "run.py")
    if root is None:
        print("Experiment commands require an equiv_dens_ml checkout.", file=sys.stderr)
        return 2
    campaign, registry = load_registry(root)
    if args.action == "list":
        for name, task in sorted(registry.items()):
            print(f"{name}\t{task['kind']}")
        return 0
    if not args.task or args.task not in registry:
        print(f"Unknown JACS task: {args.task!r}; use 'experiment jacs list'.", file=sys.stderr)
        return 2
    task = registry[args.task]
    remaining = list(args.extra or [])
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if args.action == "submit":
        if task["kind"] != "slurm":
            print("Only Slurm tasks support submit.", file=sys.stderr)
            return 2
        command = ["sbatch", *remaining, str(campaign / task["target"])]
    elif task["kind"] == "python":
        command = [sys.executable, "-m", task["target"], *remaining]
    elif task["kind"] == "shell":
        command = ["bash", str(campaign / task["target"]), *remaining]
    else:
        print("Use submit for a Slurm task.", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    try:
        return subprocess.call(command, cwd=root, env=env)
    except FileNotFoundError as exc:
        print(f"Cannot launch task: {exc}", file=sys.stderr)
        return 127
