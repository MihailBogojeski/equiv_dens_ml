"""Source-isolated, untimed H5-E1 energy/force numerical diagnostics.

Use the shared pipeline/paired_pipeline CLI for coordinator-owned timings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.fixtures import load_fixture, select_source
from benchmarks import pipeline


def energy_snapshot(fixture, device="cpu", finite_differences=False,
                    fd_steps=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)):
    result = {"fixture_hash": fixture["fixture_hash"], "state_hash": fixture["state_hash"],
              "contracts": {}, "exclusions": {}, "successive_frames": []}
    for mode, training in (("ef", False), ("ef", True), ("force_training", True)):
        key = f"{mode}/{'train' if training else 'eval'}"
        result["contracts"][key] = pipeline.evaluate_contract(fixture, mode, training, device)
    if finite_differences:
        result["finite_differences"] = pipeline.finite_difference_checks(fixture, device, fd_steps)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("legacy", "device"), default="legacy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--finite-differences", action="store_true")
    parser.add_argument("--fd-steps", type=float, nargs="+",
                        default=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6))
    args = parser.parse_args()
    root = select_source(args.source_root)
    from benchmarks.precision import use_full_precision
    use_full_precision()
    fixture = load_fixture(args.fixture)
    pipeline.ENERGY_BACKEND = args.backend
    result = energy_snapshot(fixture, args.device, args.finite_differences, args.fd_steps)
    result["provenance"] = pipeline.provenance(root, fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    if args.finite_differences:
        for key in ("force_fd_pass", "force_loss_mixed_fd_pass", "spatial_fd_pass"):
            if not result["finite_differences"][key]:
                raise AssertionError(f"Independent finite difference failed: {key}; raw record: {args.output}")


if __name__ == "__main__":
    main()
