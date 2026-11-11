#!/usr/bin/env python3
"""Shared CLI defaults for H-bond geometry generators."""

from __future__ import annotations

import argparse

QUICK_TRAIN = 20
QUICK_VAL = 4
QUICK_ID = 4
QUICK_OOD = 10
QUICK_STEPS = 10
QUICK_EQUIL = 50
QUICK_MAX_SWEEPS = 8
QUICK_POOL = 80
QUICK_N_EMBED = 20


def add_quick_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small frame counts for pipeline smoke tests (gates still run)",
    )


def apply_quick(args: argparse.Namespace) -> None:
    if not getattr(args, "quick", False):
        return
    if hasattr(args, "n_train"):
        args.n_train = QUICK_TRAIN
    if hasattr(args, "n_val"):
        args.n_val = QUICK_VAL
    if hasattr(args, "n_id"):
        args.n_id = QUICK_ID
    if hasattr(args, "n_ood"):
        args.n_ood = QUICK_OOD
    if hasattr(args, "steps_between"):
        args.steps_between = QUICK_STEPS
    if hasattr(args, "equilibration_steps"):
        args.equilibration_steps = QUICK_EQUIL
    if hasattr(args, "max_sweeps"):
        args.max_sweeps = QUICK_MAX_SWEEPS
    if hasattr(args, "pool_size"):
        args.pool_size = QUICK_POOL
    if hasattr(args, "n_embed"):
        args.n_embed = QUICK_N_EMBED
