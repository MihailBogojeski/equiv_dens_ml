#!/usr/bin/env python3
"""Launch one or more energy-stage trainings that share a GPU allocation."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/run_energy_pack.py'


import argparse
import os
import subprocess
import sys
from pathlib import Path

_LIB = _LEGACY_FILE.resolve().parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from experiments.jacs_revision.lib.energy_pack import run_pack  # noqa: E402
from experiments.jacs_revision.lib.energy_train_spec import (  # noqa: E402
    build_dry_run_argv,
    build_train_argv,
    campaign_name,
    choose_resume,
    density_parent,
    density_run_dir,
    mark_training_done,
    parse_spec,
    should_submit_pack,
    spec_is_done,
)
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402


class _Immediate:
    def __init__(self, code: int) -> None:
        self._code = code

    def poll(self) -> int:
        return self._code


def _launch_train(
    spec: str,
    repo: Path,
    *,
    dry_run: bool,
    force_new: bool,
) -> object:
    if dry_run:
        print("dry-run", " ".join(build_dry_run_argv(spec, repo)))
        return _Immediate(0)
    if not (density_parent(repo, spec) / ".density_training_done").is_file():
        print(f"{spec}: missing .density_training_done", file=sys.stderr)
        return _Immediate(1)
    density = density_run_dir(repo, spec)
    if density is None:
        print(f"{spec}: density run directory missing", file=sys.stderr)
        return _Immediate(1)
    resume = choose_resume(repo, spec, force_new=force_new)
    if resume is not None:
        print(f"resuming energy training for {campaign_name(*parse_spec(spec))} from {resume}")
    argv = build_train_argv(
        spec,
        repo,
        density_run=str(density),
        resume=str(resume) if resume is not None else None,
    )
    name = campaign_name(*parse_spec(spec))
    log_path = repo / "logs" / f"train_energy_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")
    print(f"launching {spec} -> {log_path}")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("WANDB_MODE", "offline")
    return subprocess.Popen(
        argv,
        cwd=str(repo),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--specs", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--check-submit", action="store_true")
    parser.add_argument("--queued-names", nargs="*", default=[])
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo) if args.repo is not None else repo_root()

    if args.check_submit:
        return 0 if should_submit_pack(args.specs, args.queued_names) else 1

    if not args.specs:
        parser.error("--specs is required unless --check-submit")

    def is_done(spec: str) -> bool:
        return spec_is_done(repo, spec)

    def launch(spec: str) -> object:
        return _launch_train(
            spec, repo, dry_run=args.dry_run, force_new=args.force_new
        )

    results = run_pack(
        args.specs,
        args.slots,
        is_done=is_done,
        launch=launch,
    )
    status = 0
    for spec, code in results:
        if code == 0:
            if not args.dry_run:
                mark_training_done(repo, spec)
            print(f"{spec} energy stage finished")
        else:
            print(f"{spec} exited {code}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
