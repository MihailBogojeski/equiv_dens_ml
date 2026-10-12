"""Read installed metadata without importing GPU libraries or starting a model."""

import importlib.metadata as metadata
import json
import platform
import sys


PACKAGES = (
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "schnetpack",
    "pyscf", "gpu4pyscf-cuda12x", "cupy-cuda12x", "cutensor-cu12", "dftd4",
    "dftpy", "ase", "rdkit", "e3nn", "mace-torch", "cuequivariance",
    "cuequivariance-torch", "cuequivariance-ops-torch-cu12", "pytest",
    "pytest-timeout", "apsw", "tensorboardX", "wandb", "h5py", "memspectrum",
    "jupyterlab", "ruff", "jax", "jaxlib", "jax-cuda12-plugin",
    "jax-cuda12-pjrt", "flax", "optax", "orbax-checkpoint", "jax-md", "jraph",
    "e3nn-jax", "cuequivariance-jax", "cuequivariance-ops-jax-cu12",
)


def main():
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    print(json.dumps({"python": sys.version, "executable": sys.executable,
                      "platform": platform.platform(), "packages": packages},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
