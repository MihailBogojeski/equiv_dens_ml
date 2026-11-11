"""Select an existing training workflow without altering its arguments."""
import argparse
from importlib import import_module

WORKFLOWS = {
    "density": "equiv_dens.training.workflows.density",
    "joint": "equiv_dens.training.workflows.joint",
    "multiphase": "equiv_dens.training.workflows.multiphase",
    "energy": "equiv_dens.training.workflows.energy",
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=WORKFLOWS, default="joint")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return import_module(WORKFLOWS[args.mode]).main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
