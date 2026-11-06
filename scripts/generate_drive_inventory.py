#!/usr/bin/env python3
"""Generate docs/drive_archive_inventory.md from imported archive contents."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "paper" / "_staging"
ARCHIVES = REPO_ROOT / "paper" / "archives"
MODELS = REPO_ROOT / "paper" / "models"
TRAJ = REPO_ROOT / "paper" / "trajectories"
MANUSCRIPTS = REPO_ROOT / "docs" / "manuscripts"
OUTPUT = REPO_ROOT / "docs" / "drive_archive_inventory.md"


def du_sh(path: Path) -> str:
    if not path.exists():
        return "0"
    r = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=False)
    return r.stdout.split()[0] if r.stdout else "?"


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def fmt_size(n: int | None) -> str:
    if n is None:
        return "missing"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def inspect_npy_dict(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == ():
            obj = arr.item()
            if isinstance(obj, dict):
                parts = [f"keys={list(obj.keys())}"]
                for k, v in list(obj.items())[:5]:
                    if hasattr(v, "shape"):
                        parts.append(f"{k}: shape={v.shape}, dtype={v.dtype}")
                return "; ".join(parts)
        return inspect_npy(path)
    except Exception as exc:
        return f"error: {exc}"


def inspect_npy(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            return f"shape={arr.shape}, dtype={arr.dtype}"
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = list(arr.files)
            parts = [f"keys={keys}"]
            for k in keys[:5]:
                a = arr[k]
                if hasattr(a, "shape"):
                    parts.append(f"{k}: shape={a.shape}, dtype={a.dtype}")
            return "; ".join(parts)
        return f"type={type(arr)}"
    except Exception as exc:
        return f"error: {exc}"


def zip_members(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    r = subprocess.run(["zipinfo", "-1", str(path)], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        r = subprocess.run(["unzip", "-l", str(path)], capture_output=True, text=True, check=False)
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("---")]
    return lines[:max_lines]


def pdf_pages(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not shutil.which("pdfinfo"):
        return "n/a (pdfinfo not installed)"
    r = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return "pdfinfo unavailable"
    for line in r.stdout.splitlines():
        if line.startswith("Pages:"):
            return line.split(":", 1)[1].strip()
    return "?"


def diff_text(a: Path, b: Path) -> str:
    if not a.exists() or not b.exists():
        return "one or both missing"
    r = subprocess.run(["diff", "-q", str(a), str(b)], capture_output=True, text=True, check=False)
    return "identical" if r.returncode == 0 else "different"


def load_gmail_status() -> dict:
    p = STAGING / "gmail" / "recovery_status.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> int:
    gmail_status = load_gmail_status()
    lines: list[str] = [
        "# Drive Archive Import Inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Disk totals",
        "",
        f"| Path | Size |",
        f"|------|------|",
        f"| `paper/models/` | {du_sh(MODELS)} |",
        f"| `paper/archives/` | {du_sh(ARCHIVES)} |",
        f"| `paper/trajectories/` | {du_sh(TRAJ)} |",
        f"| `docs/manuscripts/` | {du_sh(MANUSCRIPTS)} |",
        f"| `paper/_staging/download/` | {du_sh(STAGING / 'download')} |",
        "",
        "## Archive zips",
        "",
        "| File | Size | Top members (first 10) | Status |",
        "|------|------|------------------------|--------|",
    ]

    zip_sources = list(ARCHIVES.glob("*.zip")) if ARCHIVES.exists() else []
    if not zip_sources:
        zip_sources = list((STAGING / "download").rglob("*.zip"))

    seen: set[str] = set()
    for zp in sorted(zip_sources):
        name = zp.name
        if name in seen:
            continue
        seen.add(name)
        members = zip_members(zp, 10)
        member_str = "<br>".join(members[:10]) if members else "(empty or unreadable)"
        pth_hits = [m for m in zip_members(zp, 500) if m.endswith((".pth", ".pt"))]
        if pth_hits:
            member_str += "<br>..." + f" **checkpoints:** {', '.join(pth_hits[:5])}"
        status = "archived"
        if (MODELS / "_unmapped" / name.replace(".zip", "")).exists():
            status = "extracted -> _unmapped"
        elif name.startswith("thiophene_md") or name.startswith("thiophene_traj"):
            status = "extracted -> trajectories"
        elif (MODELS / "polythiophene").exists() and "thiophene" in name and "traj" not in name:
            status = "extracted -> models"
        lines.append(f"| `{name}` | {fmt_size(file_size(zp))} | {member_str} | {status} |")

    lines.extend(["", "## Trajectory files", "", "| File | Location | Size | Inspection | Overlap |", "|------|----------|------|------------|---------|"])

    traj_targets = [
        ("simulation_local_all_8mer_md_001_0.xyz", TRAJ / "polythiophene/8mer"),
        ("simulation_resorcinol_cluster_all_001_compressed_0.npy", TRAJ / "resorcinol"),
        ("simulation_resorcinol_cluster_all_001_compressed_1.npy", TRAJ / "resorcinol"),
        ("simulation_resorcinol_cluster_all_001_compressed_2.npy", TRAJ / "resorcinol"),
        ("simulation_resorcinol_cluster_all_001_compressed_3.npy", TRAJ / "resorcinol"),
        ("simulation_cluster_all_001_compressed_0.npy", TRAJ / "ethanol"),
        ("simulation_-ethanethiol_cluster_all_001_compressed_0.npy", TRAJ / "ethanethiol"),
        ("loss_base.csv", TRAJ / "ethanol"),
        ("loss_delta.csv", TRAJ / "ethanol"),
        ("thiophene4mer_test_d4.npy", TRAJ / "polythiophene"),
        ("thiophene6mer_test_d4.npy", TRAJ / "polythiophene"),
    ]

    for fname, loc in traj_targets:
        found = loc / fname if loc.exists() else None
        if found is None or not found.exists():
            # search staging
            hits = list((STAGING / "download").rglob(fname))
            found = hits[0] if hits else Path("(missing)")
        size = fmt_size(file_size(found) if found.exists() else None)
        insp = inspect_npy_dict(found) if fname.endswith(".npy") else ("present" if found.exists() else "missing")
        ds_hit = REPO_ROOT / "datasets" / fname
        overlap = "already present in datasets/" if ds_hit.exists() else ("new" if found.exists() else "gmail_manual")
        lines.append(f"| `{fname}` | `{loc.relative_to(REPO_ROOT) if loc.exists() else '?'}` | {size} | {insp} | {overlap} |")

    lines.extend(["", "## Config / notebook diffs", ""])
    config_pairs = [
        (
            REPO_ROOT / "config/paper/thiophene_poly_all_001_coreless_dpm.txt",
            MANUSCRIPTS / "source_configs/thiophene_poly_all_001_coreless_dpm.drive_archive.txt",
            STAGING / "download",
            "thiophene_poly_all_001_coreless_dpm.txt",
        ),
        (
            REPO_ROOT / "paper/models/args/ethanethiol_all_004_31_coreless.txt",
            MANUSCRIPTS / "source_configs/ethanethiol_all_004_31_coreless.drive_archive.txt",
            STAGING / "download",
            "ethanethiol_all_004_31_coreless.txt",
        ),
    ]
    for tracked, archive_copy, search_root, fname in config_pairs:
        if not archive_copy.exists():
            hits = list(search_root.rglob(fname)) if search_root.exists() else []
            if hits:
                archive_copy.parent.mkdir(parents=True, exist_ok=True)
                archive_copy.write_bytes(hits[0].read_bytes())
        result = diff_text(tracked, archive_copy) if archive_copy.exists() else "archive copy not found"
        lines.append(f"- `{tracked.name}` vs Drive/Gmail copy: **{result}**")

    nb_tracked = REPO_ROOT / "paper/notebooks/thiophene_append_clean.ipynb"
    lines.append(f"- `thiophene_append_clean.ipynb`: {'present' if nb_tracked.exists() else 'missing (check Drive archive 03_Manual or Gmail)'}")

    lines.extend(["", "## Gmail recovery status", ""])
    if gmail_status.get("recovered"):
        for item in gmail_status["recovered"]:
            lines.append(f"- RECOVERED `{item['file']}` via {item['method']} -> `{item['dest']}`")
    if gmail_status.get("manual"):
        lines.append("")
        lines.append("| File | Message ID | Size | Status | Notes |")
        lines.append("|------|------------|------|--------|-------|")
        for item in gmail_status["manual"]:
            fname = item["file"]
            recovered_elsewhere = False
            notes = item.get("drop_path", "")
            for search_root in [TRAJ, MANUSCRIPTS, REPO_ROOT / "paper" / "notebooks", STAGING / "download"]:
                if list(search_root.rglob(fname)) if search_root.exists() else []:
                    recovered_elsewhere = True
                    notes = f"recovered via Drive archive (not Gmail API)"
                    break
            status = "recovered (Drive archive)" if recovered_elsewhere else "gmail_manual"
            lines.append(
                f"| `{fname}` | `{item.get('msg_id', '')}` | {item.get('size_mb', '?')} MB | {status} | {notes} |"
            )

    lines.extend(["", "## Standalone Drive files (gdown blocked — manual download)", ""])
    standalone_failed = [
        ("new_energy_models.zip", "16vtZsd7ra8DFC7ICIBZAWJioXUZcLOL7", "3.3 GB", STAGING / "download" / "new_energy_models.zip"),
        ("thiophene_traj_with_velocities.zip", "1bLz2lafxPV88HwcocZ_Ec-pnWVF41X21", "4.0 GB", STAGING / "download" / "thiophene_traj_with_velocities.zip"),
        ("simulation_resorcinol_cluster_all_001_compressed_2.npy", "1h54CKkXMp07tZD4hjdh-9fXd64T2DVNI", "57 MB", TRAJ / "resorcinol/simulation_resorcinol_cluster_all_001_compressed_2.npy"),
        ("simulation_resorcinol_cluster_all_001_compressed_3.npy", "1QHVRXpZyhdOs6SrKHoPCrqbJ8WGtQGEl", "57 MB", TRAJ / "resorcinol/simulation_resorcinol_cluster_all_001_compressed_3.npy"),
    ]
    lines.append("| File | Drive ID | Expected size | Local status |")
    lines.append("|------|----------|---------------|--------------|")
    for name, fid, size, dest in standalone_failed:
        status = "present" if dest.exists() else "gdown permission denied — drop in docs/manuscripts/gmail_manual/"
        lines.append(f"| `{name}` | `{fid}` | {size} | {status} |")

    unmapped = MODELS / "_unmapped"
    if unmapped.exists():
        lines.extend(["", "## Unmapped model extractions (`paper/models/_unmapped/`)", ""])
        for d in sorted(unmapped.iterdir()):
            if d.is_dir():
                lines.append(f"- `{d.name}/` — {du_sh(d)}")

    lines.extend(["", "## NPY dict keys (allow_pickle cluster trajectories)", ""])
    for npy_path in sorted(TRAJ.rglob("simulation_*compressed*.npy")):
        insp = inspect_npy_dict(npy_path)
        rel = npy_path.relative_to(REPO_ROOT)
        lines.append(f"- `{rel}`: {insp}")

    lines.extend(["", "## Manuscripts recovered", ""])
    if MANUSCRIPTS.exists():
        pdfs = sorted(MANUSCRIPTS.rglob("*.pdf"))
        drive_pdfs = sorted((STAGING / "download").rglob("*.pdf")) if (STAGING / "download").exists() else []
        all_pdfs = {p.name: p for p in drive_pdfs}
        for p in pdfs:
            all_pdfs[p.name] = p
        for name, p in sorted(all_pdfs.items())[:40]:
            rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
            lines.append(f"- `{rel}` — {fmt_size(file_size(p))}, pages={pdf_pages(p)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
