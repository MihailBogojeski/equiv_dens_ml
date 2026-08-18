#!/usr/bin/env python3
"""Turn one shard of CSH Fock matrices into density-fitting coefficients.

Per structure: apply the signed AO permutation, solve F C = S C eps with the
overlap rebuilt in PySCF, form P = 2 C_occ C_occ^T, and project onto the
auxiliary basis DenSNet trains on.

Shards are claimed with an exclusive-create lock and marked done on completion,
so CPU and GPU array jobs can drain the same shard list at the same time
without duplicating work and without coordination. Rerunning is a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle  # noqa: E402
from omol_csh_density_fit import fit_density  # noqa: E402
from omol_csh_validate_ordering import build_transform, load_table  # noqa: E402

HARTREE_TO_EV = 27.211386245988


def solve_density(fock, ovlp, nelec, s_thresh, device):
    """Closed-shell density from the converged Fock matrix.

    def2-TZVPD is diffuse enough that the overlap is numerically singular for
    these system sizes, so near-null directions are projected out before
    diagonalising (HELM filter the same way).
    """
    if device == "gpu":
        import torch

        dev = "cuda"
        s_t = torch.as_tensor(ovlp, device=dev, dtype=torch.float64)
        f_t = torch.as_tensor(fock, device=dev, dtype=torch.float64)
        s_val, s_vec = torch.linalg.eigh(s_t)
        keep = s_val > s_thresh
        x = s_vec[:, keep] / torch.sqrt(s_val[keep])
        eps, c = torch.linalg.eigh(x.T @ f_t @ x)
        coeff = (x @ c).cpu().numpy()
        eps = eps.cpu().numpy()
        n_kept = int(keep.sum().item())
    else:
        s_val, s_vec = np.linalg.eigh(ovlp)
        keep = s_val > s_thresh
        x = s_vec[:, keep] / np.sqrt(s_val[keep])
        eps, c = np.linalg.eigh(x.T @ fock @ x)
        coeff = x @ c
        n_kept = int(keep.sum())

    nocc = nelec // 2
    occ = coeff[:, :nocc]
    return 2.0 * occ @ occ.T, eps, n_kept


def process(entry, handle, table, args, device):
    node = handle[entry["path"]]
    elements = np.asarray(node["elements"][()])
    coords = np.asarray(node["coords"][()])
    fock_flat = np.asarray(node["fock"][()])

    mol = build_mol(elements, coords, entry["charge"], 0)
    perm, signs = build_transform(mol, elements, table)
    fock = inflate_triangle(fock_flat)[np.ix_(perm, perm)] * np.outer(signs, signs)
    ovlp = mol.intor("int1e_ovlp")

    dm, eps, n_kept = solve_density(fock, ovlp, mol.nelectron, args.s_thresh, device)

    coeffs, auxmol, info = fit_density(
        mol,
        dm,
        args.auxbasis,
        block=args.block,
        device=device,
        gpu_budget_bytes=args.gpu_budget,
    )

    nocc = mol.nelectron // 2
    return {
        "path": entry["path"],
        "positions": coords.astype(np.float64),
        "atom_numbers": elements.astype(np.int64),
        "df_coeff": coeffs.astype(np.float64),
        "charge": int(entry["charge"]),
        "n_elec": int(mol.nelectron),
        "nao": int(mol.nao),
        "naux": int(info["naux"]),
        "n_elec_df": float(info["n_elec_df"]),
        "n_elec_err": float(info["n_elec_error"]),
        "eps_min": float(eps[0]),
        "homo": float(eps[nocc - 1]),
        "gap_ev": float((eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV),
        "n_kept": n_kept,
        "used_gpu": bool(info["used_gpu"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--table", default="datasets/revision/omol_csh/ao_permutation.json")
    parser.add_argument("--auxbasis", default="augccpvqzjkfit")
    parser.add_argument("--s-thresh", type=float, default=1e-5)
    parser.add_argument("--block", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=("cpu", "gpu", "auto"))
    parser.add_argument("--gpu-budget", type=int, default=8_000_000_000)
    parser.add_argument("--max-atoms", type=int, default=0, help="0 = no cap")
    parser.add_argument("--n-elec-tol", type=float, default=1e-3)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest["entries"]
    if args.max_atoms:
        entries = [e for e in entries if e["natoms"] <= args.max_atoms]

    start = args.shard * args.shard_size
    chunk = entries[start : start + args.shard_size]
    if not chunk:
        print(f"shard {args.shard}: empty, nothing to do")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / f"shard_{args.shard:05d}.done"
    claim = outdir / f"shard_{args.shard:05d}.claim"
    out = outdir / f"shard_{args.shard:05d}.npz"

    if done.exists():
        print(f"shard {args.shard}: already done")
        return 0
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} {os.getpid()} {device}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"shard {args.shard}: claimed by another worker")
        return 0

    print(f"shard {args.shard}: {len(chunk)} structures on {device}", flush=True)
    records, failures = [], []
    began = time.time()
    try:
        with h5py.File(args.h5, "r") as handle:
            for i, entry in enumerate(chunk):
                t0 = time.time()
                try:
                    rec = process(entry, handle, table_cache(args.table), args, device)
                except Exception as exc:  # keep the shard alive; log and continue
                    failures.append({"path": entry["path"], "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  [{i+1}/{len(chunk)}] FAIL {entry['path']}: {exc}", flush=True)
                    continue
                if abs(rec["n_elec_err"]) > args.n_elec_tol:
                    failures.append(
                        {"path": entry["path"], "error": f"n_elec drift {rec['n_elec_err']:.2e}"}
                    )
                    print(f"  [{i+1}/{len(chunk)}] REJECT {entry['path']}", flush=True)
                    continue
                records.append(rec)
                print(
                    f"  [{i+1}/{len(chunk)}] {entry['path'].split('/')[-1][:38]:<38} "
                    f"nao={rec['nao']:<5} naux={rec['naux']:<5} "
                    f"dNe={rec['n_elec_err']:+.1e} {time.time()-t0:.1f}s",
                    flush=True,
                )

        np.savez_compressed(
            out,
            records=np.array(records, dtype=object),
            failures=np.array(failures, dtype=object),
            allow_pickle=True,
        )
        done.write_text(
            json.dumps(
                {
                    "shard": args.shard,
                    "n_ok": len(records),
                    "n_failed": len(failures),
                    "device": device,
                    "seconds": time.time() - began,
                }
            )
        )
        print(
            f"shard {args.shard}: {len(records)} ok, {len(failures)} failed, "
            f"{time.time()-began:.1f}s -> {out}"
        )
    finally:
        claim.unlink(missing_ok=True)
    return 0


_TABLE = {}


def table_cache(path):
    if path not in _TABLE:
        _TABLE[path] = load_table(path)
    return _TABLE[path]


if __name__ == "__main__":
    sys.exit(main())
