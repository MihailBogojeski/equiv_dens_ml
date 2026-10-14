# DenSNet runtimes

Separate Torch and JAX images cover the current pipeline and the planned
framework/backbone investigations. The images contain dependencies; the live
project checkout supplies scientific source.

The recipes extend the identified local base images, rather than a moving Docker
tag. Core constraints and locked requirements in this directory pin package
versions; provenance manifests identify the packaged third-party artifacts.

## Use

Run from the project root inside a compute allocation:

```bash
tools/runtime/run.sh torch python run.py --help
tools/runtime/run.sh torch python -m pytest tests/ -q
tools/runtime/run.sh torch python tools/runtime/smoke_torch.py --device cpu
tools/runtime/run.sh jax python tools/runtime/smoke_jax.py --device cpu
```

On a GPU allocation, use `--device cuda` for both qualification scripts.

The JAX image runs core JAX, Flax, Optax, JAX-MD and e3nn-jax on CPU-only nodes.
Its CUDA-enabled cuEquivariance JAX frontend eagerly loads NVIDIA driver
libraries; use that frontend on a GPU node with `--nv`. CPU qualification
explicitly excludes this frontend, and CUDA qualification requires its native
execution and derivatives.

The wrapper enables NVIDIA binds when the host has `nvidia-smi`, clears inherited
Apptainer binds, starts a clean environment, preserves Slurm's GPU visibility,
sets project import paths, and caps thread/compile workers to allocated CPUs.
`DENSNET_TORCH_IMAGE`, `DENSNET_JAX_IMAGE`, `DENSNET_NUM_THREADS` and
`EQUIV_DENS_PROJECT_ROOT` override those choices. For heavy repeated use, stage the
image to node-local storage and supply its path through the image override; the
shared `container_exec` adapter also supplies atomic staging when configured.

Torch includes CPU/GPU PySCF, the supported CuPy/cuTENSOR pair, D4, ASE,
SchNetPack, RDKit, IR dependencies, e3nn/MACE/cuEquivariance, tests, profiling,
notebooks, and CUDA/C++ compilers. JAX includes JAX CUDA12, Flax, Optax, Orbax,
JAX-MD, Jraph, e3nn-jax, cuEquivariance, PySCF and development tools. The native
pipeline extension (`jax-pipeline.def`) also pins ASE3.29.0 and DFTpy2.2.0 for
Torch-free host preparation/MD. All previous locked versions remain unchanged.
Its installed image is `/home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif`;
the wrapper prefers it when present. The existing image remains available for
comparison. Build the extension with `sbatch tools/runtime/build_jax_pipeline.sbatch`.

Legacy g-xTB v1.1.0 is pinned with asset checksums and matches this project's
binary/parameter interface. It is not a claim of the paper's reference version.
For existing g-xTB configuration files, override the checkout-relative paths:

```bash
tools/runtime/run.sh torch python run.py gxtb-md \
  @config/md/gxtb/gxtb_polythiophene_2mer.txt \
  --gxtb-path=/opt/gxtb/bin/gxtb --gxtb-params=/opt/gxtb/parameters
```

Model checkpoints, authoritative datasets, W&B credentials and project-specific
reference fixtures are external inputs, not container dependencies.
Trained checkpoints and labeled datasets are not required to begin performance
profiling and numerical equivalence checks with reproducible initialized weights
and valid generated inputs. The existing checkpoint-specific tests are optional
for that phase and retain explicit missing-checkpoint skips.

## Rebuild

The definitions' `From:` paths must identify the recorded base images. Keep
Torch and JAX separate. Build from the repository root on a CPU allocation:

```bash
mkdir -p /temp/$(id -un)/densnet-build/cache
env -u APPTAINER_BIND -u APPTAINER_BINDPATH \
  -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  APPTAINER_TMPDIR=/temp/$(id -un)/densnet-build \
  APPTAINER_CACHEDIR=/temp/$(id -un)/densnet-build/cache \
  GOMAXPROCS=1 apptainer build --fakeroot \
  --mksquashfs-args '-comp zstd -Xcompression-level 3 -mem 1500M -processors 2' \
  /temp/$(id -un)/densnet-build/densnet-torch.sif containers/torch.def
```

Use `containers/jax.def` and a different output path for JAX. The 1500 MB/two
processor cap is appropriate to the audited eight-GB/two-CPU build allocation;
adjust deliberately for a different allocation. The installed Apptainer1.5.2
mounts these zstd images; check support when transferring to older installations.
Keep `--fakeroot` when packaging a previously built sandbox too, so subordinate
UID-owned files are readable and included. Build and qualification failures
must be resolved before publishing a replacement image. Never overwrite a
reference image implicitly.

SchNetPack 2.1.1 caps SymPy at 1.12, conflicting with modern Torch. The included
`2.1.1+densnet1` wheel changes only the version/dependency metadata and RECORD;
its 142 package files are byte-identical to the original. Recreate it with
`tools/runtime/prepare_schnetpack_wheel.py` from the upstream wheel, and verify
against `schnetpack-wheel-provenance.json`. The project's existing `equiv_dens.compat`
must still precede SchNetPack imports. SchNetPack 2.2 requires Python 3.12 and was
not substituted into the Python 3.11 Torch stack.

`*.in` declares intended packages, core constraints preserve audited versions,
and `*.lock` pins the dependency closure. Torch resolution/install deliberately
searches both PyPI and the PyTorch CUDA126 index using uv's
`unsafe-best-match` strategy: default first-index precedence selected obsolete
shared dependencies from the PyTorch index. Both indexes are explicit; installed
versions remain pinned. Each build runs `pip check` and stores the full effective
manifest at `/opt/densnet-runtime/installed.txt`, including inherited packages.

The CUDA compiler is supplied by the Torch base at `/usr/local/cuda/bin/nvcc`.
The smoke and qualification scripts exercise the installed dependencies.
Custom operators and new backbones need numerical/derivative checks as well.
