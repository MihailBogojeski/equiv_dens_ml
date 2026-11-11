"""Render representative revision geometries with open-source PyMOL."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/render_dataset_structures.py'


import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, response_figure_path, save_response_figure

_REPO = _LEGACY_FILE.resolve().parents[3]
_DEFAULT_OUT = _REPO / "results" / "revision" / "figures" / "response"

import sys

sys.path.insert(0, str(_REPO / "scripts" / "revision" / "lib"))
from experiments.jacs_revision.lib.geoopt_metrics import align_onto  # noqa: E402

_DELTA = re.compile(r"delta=([+-]?(?:\d+\.?\d*|\.\d+))")
MALON_TRAIN_DELTA_MAX = -0.40


@dataclass(frozen=True)
class Frame:
    symbols: list[str]
    coords: list[list[float]]
    comment: str


@dataclass(frozen=True)
class Panel:
    row: str
    label: str
    source: Path
    kind: str = "file"
    target: float | int | None = None


def n_water(symbols: list[str]) -> int:
    return sum(1 for s in symbols if s in {"O", "8", "o"})


def heavy_min_distance(frame: Frame) -> float:
    """Shortest non-hydrogen contact (Å). Fused dimers land well below 1.1."""
    coords = []
    for symbol, xyz in zip(frame.symbols, frame.coords):
        if symbol.upper() in {"H", "D", "T"}:
            continue
        coords.append(xyz)
    if len(coords) < 2:
        return float("inf")
    best = float("inf")
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
            if dist < best:
                best = dist
    return best


def select_first_clean(frames: list[Frame], min_heavy: float = 1.15) -> Frame:
    """Skip interpenetrating / exploded --quick frames."""
    for frame in frames:
        if heavy_min_distance(frame) >= min_heavy:
            return frame
    raise ValueError(f"no frame with heavy-atom separation >= {min_heavy} Å")


def parse_delta(comment: str) -> float:
    match = _DELTA.search(comment)
    if match is None:
        raise ValueError(f"no delta= in comment: {comment!r}")
    return float(match.group(1))


def malon_region(delta: float, cutoff: float = MALON_TRAIN_DELTA_MAX) -> str:
    """Training occupies one enol well: delta = r(Od-H) - r(H-Oa) <= -0.40 Angstrom."""
    return "training" if delta <= cutoff else "held-out"


def read_xyz_frames(path: Path) -> list[Frame]:
    lines = Path(path).read_text().splitlines()
    frames: list[Frame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        symbols: list[str] = []
        coords: list[list[float]] = []
        for row in lines[i + 2 : i + 2 + n_atoms]:
            parts = row.split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append(Frame(symbols, coords, comment))
        i += 2 + n_atoms
    return frames


def select_first_with_n(frames: list[Frame], n: int) -> Frame:
    for frame in frames:
        if n_water(frame.symbols) == n:
            return frame
    raise ValueError(f"no frame with n={n}")


def select_mid_with_n(frames: list[Frame], n: int) -> Frame:
    """A mid-index frame of size n, so the gallery is not just the first parent."""
    matches = [frame for frame in frames if n_water(frame.symbols) == n]
    if not matches:
        raise ValueError(f"no frame with n={n}")
    return matches[len(matches) // 2]


def try_parse_delta(comment: str) -> float | None:
    match = _DELTA.search(comment)
    if match is None:
        return None
    return float(match.group(1))


def select_nearest_delta(frames: list[Frame], target: float) -> Frame:
    scored = []
    for frame in frames:
        delta = try_parse_delta(frame.comment)
        if delta is None:
            continue
        scored.append((abs(delta - target), frame))
    if not scored:
        raise ValueError(f"no frame with delta= near {target}")
    return min(scored, key=lambda item: item[0])[1]


def write_xyz(path: Path, frame: Frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(frame.symbols)}", frame.comment]
    for symbol, xyz in zip(frame.symbols, frame.coords):
        lines.append(f"{symbol:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def optional_reviewer_panels(repo: Path) -> list[Panel]:
    """NMA, MeOH–water, and AcAc when those XYZ trees have been generated."""
    rev = repo / "datasets" / "revision"
    reps = repo / "results" / "revision" / "gallery_reps"
    nma = reps / "nma_dimer.xyz"
    acac = reps / "acac.xyz"
    acac_scan = reps / "acac_scan.xyz"
    pgg = reps / "pgg.xyz"
    candidates = [
        Panel(
            "NMA dimer",
            "H-bond dimer",
            nma if nma.is_file() else rev / "nma_dimer" / "train.xyz",
            "clean" if not nma.is_file() else "file",
            1.15,
        ),
        Panel("MeOH–water", "isomer A", rev / "meoh_water" / "train.xyz", "index", 0),
        Panel(
            "AcAc",
            r"enol chelate",
            acac if acac.is_file() else rev / "acac" / "train.xyz",
            "file" if acac.is_file() else "clean",
            1.15,
        ),
        Panel(
            "AcAc",
            r"scan $\delta\approx 0$",
            acac_scan if acac_scan.is_file() else rev / "acac" / "ood_proton_transfer.xyz",
            "file" if acac_scan.is_file() else "delta",
            0.0,
        ),
        Panel("Paracetamol", "train", rev / "paracetamol" / "train.xyz", "index", 0),
        Panel(
            "PGG",
            "Phe-Gly-Gly",
            pgg if pgg.is_file() else rev / "pgg" / "train.xyz",
            "file" if pgg.is_file() else "clean",
            1.15,
        ),
    ]
    kept: list[Panel] = []
    for panel in candidates:
        if not panel.source.is_file():
            continue
        frames = read_xyz_frames(panel.source)
        if panel.kind == "clean":
            try:
                select_first_clean(frames, min_heavy=float(panel.target or 1.15))
            except ValueError:
                continue
        if panel.row == "AcAc" and "gallery_reps" not in panel.source.as_posix():
            deltas = [try_parse_delta(frame.comment) for frame in frames]
            if not any(d is not None and -1.8 <= d <= 0.15 for d in deltas):
                continue
        kept.append(panel)
    return kept


def gallery_panels(repo: Path) -> list[Panel]:
    water = repo / "datasets" / "revision" / "water_clusters"
    ood = repo / "datasets" / "revision" / "water_ood"
    malon = repo / "datasets" / "revision" / "malonaldehyde"
    rackers = repo / "datasets" / "revision" / "external" / "rackers_water" / "raw.xyz"
    panels = [
        Panel("Training", r"$n=2$ thermal", water / "train.xyz", "mid_n", 2),
        Panel("Training", r"$n=4$ thermal", water / "train.xyz", "mid_n", 4),
        Panel("Training", r"$n=6$ thermal", water / "train.xyz", "mid_n", 6),
        Panel("Size hold-out", r"$n=8$ thermal", water / "ood_size.xyz", "mid_n", 8),
        Panel("Size hold-out", r"$n=10$ thermal", water / "ood_size.xyz", "mid_n", 10),
        Panel("Size hold-out", r"$n=12$ thermal", water / "ood_size.xyz", "mid_n", 12),
        Panel("Ice", r"$n=12$ ice Ih", ood / "ood_order.xyz", "mid_n", 12),
        Panel("Ice", r"$n=24$ ice Ih", ood / "ood_order.xyz", "mid_n", 24),
        Panel("Droplets", r"$n=16$ droplet", ood / "ood_density.xyz", "mid_n", 16),
        Panel("Droplets", r"$n=24$ droplet", ood / "ood_density.xyz", "mid_n", 24),
        Panel("Rackers", r"$n=8$", rackers, "mid_n", 8),
        Panel("Rackers", r"$n=15$", rackers, "mid_n", 15),
        Panel("Rackers", r"$n=30$", rackers, "mid_n", 30),
        Panel("Malonaldehyde", r"train enol $\delta\leq-0.40\,\mathrm{\AA}$", malon / "train.xyz", "index", 0),
        Panel("Malonaldehyde", r"scan $\delta\approx 0$", malon / "ood_proton_transfer.xyz", "delta", 0.0),
        Panel("Malonaldehyde", r"scan $\delta>0$", malon / "ood_proton_transfer.xyz", "delta", 0.58),
    ]
    return panels + optional_reviewer_panels(repo)


def resolve_frame(panel: Panel) -> Frame:
    frames = read_xyz_frames(panel.source)
    if panel.kind == "file":
        return frames[0]
    if panel.kind == "index":
        return frames[int(panel.target)]
    if panel.kind == "first_n":
        return select_first_with_n(frames, int(panel.target))
    if panel.kind == "mid_n":
        return select_mid_with_n(frames, int(panel.target))
    if panel.kind == "delta":
        return select_nearest_delta(frames, float(panel.target))
    if panel.kind == "clean":
        cutoff = 1.15 if panel.target is None else float(panel.target)
        return select_first_clean(frames, min_heavy=cutoff)
    raise ValueError(f"unknown panel kind {panel.kind}")


def find_pymol() -> str:
    for key in ("PYMOL", "PYMOL_BIN"):
        env = os.environ.get(key)
        if env and Path(env).is_file():
            return env
    local = _REPO / ".conda" / "pymol-open-source" / "bin" / "pymol"
    if local.is_file():
        return str(local)
    pixi = _REPO / ".conda" / "pymol-os" / ".pixi" / "envs" / "default" / "bin" / "pymol"
    if pixi.is_file():
        return str(pixi)
    found = shutil.which("pymol")
    if found and "incentive" not in found.lower():
        return found
    raise FileNotFoundError(
        "open-source PyMOL not on PATH; set PYMOL to pymol-open-source"
    )


def _apply_style(cmd) -> None:
    cmd.hide("everything", "mol")
    cmd.show("sticks", "mol")
    cmd.show("spheres", "mol")
    cmd.set("stick_radius", 0.13)
    cmd.set("sphere_scale", 0.20)
    cmd.do("util.cbaw")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", "on")
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", "off")
    cmd.set("specular", 0.12)
    cmd.set("ambient", 0.45)
    cmd.unset("depth_cue")
    cmd.distance("hb", "all", "all", 3.2, mode=2)
    cmd.hide("labels", "hb")
    cmd.set("dash_color", "grey70")
    cmd.set("dash_width", 1.6)
    cmd.set("dash_gap", 0.18)
    cmd.orient("mol")
    cmd.turn("x", 22)
    cmd.turn("y", 28)
    cmd.zoom("all", 1.8)


def write_overlay_pml(dft_xyz: Path, ml_xyz: Path, png: Path, zoom: float = 1.8) -> str:
    """DFT in CPK, DenSNet in teal, same stick/sphere settings as ``write_pml``."""
    return "\n".join(
        [
            "reinitialize",
            f"load {dft_xyz.as_posix()}, dft",
            f"load {ml_xyz.as_posix()}, ml",
            "hide everything",
            "show sticks, all",
            "show spheres, all",
            "set stick_radius, 0.18",
            "set sphere_scale, 0.16",
            "set valence, 0",
            "util.cbaw dft",
            "color tv_blue, ml",
            "set stick_transparency, 0.35, ml",
            "set sphere_transparency, 0.35, ml",
            "bg_color white",
            "set ray_opaque_background, on",
            "set antialias, 2",
            "set ray_shadows, off",
            "set specular, 0.12",
            "set ambient, 0.45",
            "unset depth_cue",
            "orient dft",
            "turn x, 22",
            "turn y, 28",
            f"zoom all, {zoom:.2f}",
            f"png {png.as_posix()}, width=900, height=800, dpi=200, ray=1",
            "quit",
            "",
        ]
    )


def write_pml(xyz: Path, png: Path, zoom: float = 1.8) -> str:
    symbols = [line.split()[0] for line in xyz.read_text().splitlines()[2:] if line.split()]
    water_like = set(symbols) <= {"H", "O", "h", "o"}
    hb_lines = (
        [
            "distance hb, all, all, 2.5, mode=2",
            "hide labels, hb",
            "set dash_color, grey70",
            "set dash_width, 1.6",
            "set dash_gap, 0.18",
        ]
        if water_like
        else []
    )
    return "\n".join(
        [
            "reinitialize",
            f"load {xyz.as_posix()}, mol",
            "hide everything, mol",
            "show sticks, mol",
            "show spheres, mol",
            "set stick_radius, 0.18",
            "set sphere_scale, 0.16",
            "set valence, 0",
            "util.cbaw",
            "bg_color white",
            "set ray_opaque_background, on",
            "set antialias, 2",
            "set ray_shadows, off",
            "set specular, 0.12",
            "set ambient, 0.45",
            "unset depth_cue",
            *hb_lines,
            "orient mol",
            "turn x, 22",
            "turn y, 28",
            f"zoom all, {zoom:.2f}",
            f"png {png.as_posix()}, width=800, height=800, dpi=200, ray=1",
            "quit",
            "",
        ]
    )


def _zoom_for_xyz(xyz: Path) -> float:
    n_atoms = int(xyz.read_text().splitlines()[0].strip())
    if n_atoms >= 80:
        return 2.6
    if n_atoms >= 40:
        return 2.2
    return 1.8


def render_overlay_panel(pymol: str, dft_xyz: Path, ml_xyz: Path, png: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    pml = png.with_suffix(".pml")
    zoom = min(_zoom_for_xyz(dft_xyz), _zoom_for_xyz(ml_xyz))
    pml.write_text(write_overlay_pml(dft_xyz, ml_xyz, png, zoom=zoom))
    cmd = [pymol, "-cq", str(pml)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pml.write_text(write_overlay_pml(dft_xyz, ml_xyz, png, zoom=zoom).replace("ray=1", "ray=0"))
        subprocess.run(cmd, check=True)


def read_xyz_coords(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def write_aligned_xyz(mobile_xyz: Path, reference_xyz: Path, out_xyz: Path) -> Path:
    symbols, mobile = read_xyz_coords(mobile_xyz)
    _, reference = read_xyz_coords(reference_xyz)
    aligned = align_onto(mobile, reference)
    write_xyz(
        out_xyz,
        Frame(symbols=symbols, coords=aligned.tolist(), comment="DenSNet aligned onto DFT"),
    )
    return out_xyz


def render_geoopt_comparison(
    dft_xyz: Path,
    ml_xyz: Path,
    out_dir: Path,
    stem: str = "ethanol_geoopt",
) -> Path:
    """Three-panel DFT / overlay / DenSNet figure in the gallery PyMOL style."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.pop("DISPLAY", None)
    pymol_bin = find_pymol()
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = out_dir / f"{stem}_densnet_aligned.xyz"
    write_aligned_xyz(ml_xyz, dft_xyz, aligned)
    dft_png = out_dir / f"{stem}_dft.png"
    ml_png = out_dir / f"{stem}_densnet.png"
    overlay_png = out_dir / f"{stem}_overlay.png"
    render_panel(pymol_bin, dft_xyz, dft_png)
    render_panel(pymol_bin, aligned, ml_png)
    render_overlay_panel(pymol_bin, dft_xyz, aligned, overlay_png)
    panels = [
        (Panel("Ethanol", "PBE/cc-pVDZ", dft_xyz), dft_png),
        (Panel("Ethanol", "overlay", aligned), overlay_png),
        (Panel("Ethanol", "DenSNet", aligned), ml_png),
    ]
    montage = response_figure_path(out_dir / f"{stem}.png")
    assemble_montage(panels, montage)
    return montage


def render_panel(pymol: str, xyz: Path, png: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    pml = png.with_suffix(".pml")
    zoom = _zoom_for_xyz(xyz)
    pml.write_text(write_pml(xyz, png, zoom=zoom))
    cmd = [pymol, "-cq", str(pml)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pml.write_text(write_pml(xyz, png, zoom=zoom).replace("ray=1", "ray=0"))
        subprocess.run(cmd, check=True)


LABELLED_ROWS = (
    "Training",
    "Size hold-out",
    "Ice",
    "Droplets",
    "Rackers",
    "Malonaldehyde",
)


LANDSCAPE_ROW_GROUPS = (
    ("Training", "Size hold-out", "Ice"),
    ("Droplets", "Rackers", "Malonaldehyde"),
)

_SHORT_ROW = {
    "Training": "Train",
    "Size hold-out": "Size",
    "Ice": "Ice",
    "Droplets": "Drop",
    "Rackers": "Rackers",
    "Malonaldehyde": "Malon",
}


def _landscape_title(panel: Panel) -> str:
    label = (
        panel.label.replace(" thermal", "")
        .replace(" ice Ih", "")
        .replace(" droplet", "")
        .replace(r"train enol $\delta\leq-0.40\,\mathrm{\AA}$", r"train")
        .replace(r"scan $\delta\approx 0$", r"$\delta\approx 0$")
        .replace(r"scan $\delta>0$", r"$\delta>0$")
    )
    return f"{_SHORT_ROW.get(panel.row, panel.row)} {label}"


def assemble_montage(
    panel_pngs: list[tuple[Panel, Path]],
    out: Path,
    *,
    pack: bool = False,
    n_pack_cols: int = 3,
    layout: str = "rows",
) -> None:
    apply_classic_tex()
    usetex_was = plt.rcParams["text.usetex"]
    # Word labels include unicode (Å, en-dash); mathtext CM still applies.
    plt.rcParams["text.usetex"] = False
    prefix_titles = False
    if layout == "landscape" or (pack and n_pack_cols >= 8):
        grouped: dict[str, list[tuple[Panel, Path]]] = {}
        for panel, png in panel_pngs:
            grouped.setdefault(panel.row, []).append((panel, png))
        if set(grouped) >= set(LABELLED_ROWS):
            visual_rows = []
            for group in LANDSCAPE_ROW_GROUPS:
                chunk: list[tuple[Panel, Path]] = []
                for row_name in group:
                    chunk.extend(grouped[row_name])
                visual_rows.append(("", chunk))
            n_cols = max(len(items) for _, items in visual_rows)
            n_rows = len(visual_rows)
            figsize = (14.4, 5.6)
            prefix_titles = True
        else:
            n_cols = 8
            n_rows = (len(panel_pngs) + n_cols - 1) // n_cols
            visual_rows = []
            for start in range(0, len(panel_pngs), n_cols):
                visual_rows.append(("", panel_pngs[start : start + n_cols]))
            figsize = (14.4, 5.6)
            prefix_titles = True
    elif pack:
        n_cols = n_pack_cols
        n_rows = (len(panel_pngs) + n_cols - 1) // n_cols
        visual_rows = []
        for start in range(0, len(panel_pngs), n_cols):
            chunk = panel_pngs[start : start + n_cols]
            visual_rows.append(("", chunk))
        figsize = (1.70 * n_cols + 1.2, 1.55 * n_rows + 0.35)
    else:
        row_order: list[str] = []
        grouped = {}
        for panel, png in panel_pngs:
            if panel.row not in grouped:
                row_order.append(panel.row)
                grouped[panel.row] = []
            grouped[panel.row].append((panel, png))
        if set(row_order) <= set(LABELLED_ROWS) and "Ice" in grouped and "Droplets" in grouped:
            visual_rows = [
                ("Training", grouped["Training"]),
                ("Size hold-out", grouped["Size hold-out"]),
                ("Ice / droplets", grouped["Ice"] + grouped["Droplets"]),
                ("Rackers", grouped["Rackers"]),
                ("Malonaldehyde", grouped["Malonaldehyde"]),
            ]
        else:
            visual_rows = [(row_name, grouped[row_name]) for row_name in row_order]
        n_cols = max(len(items) for _, items in visual_rows)
        n_rows = len(visual_rows)
        figsize = (1.70 * n_cols + 1.2, 1.55 * n_rows + 0.35)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
    )
    axes_arr = np.atleast_2d(axes)
    for r, (row_name, items) in enumerate(visual_rows):
        for c in range(n_cols):
            ax = axes_arr[r, c]
            ax.set_axis_off()
            if c < len(items):
                panel, png = items[c]
                if prefix_titles:
                    title = _landscape_title(panel)
                elif pack:
                    title = f"{panel.row}: {panel.label}"
                else:
                    title = panel.label
                ax.imshow(plt.imread(png))
                ax.set_title(title, fontsize=12 if prefix_titles else 11, pad=2)
            if c == 0 and row_name:
                ax.text(
                    -0.08,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=12,
                    rotation=90,
                )
    fig.tight_layout(w_pad=0.2, h_pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_response_figure(fig, out, facecolor="white")
    plt.rcParams["text.usetex"] = usetex_was


def render_gallery(repo: Path | None = None, out_dir: Path | None = None) -> Path:
    repo = repo or _REPO
    out_dir = out_dir or _DEFAULT_OUT
    panel_dir = out_dir / "structures"
    panel_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.pop("DISPLAY", None)
    pymol_bin = find_pymol()
    rendered: list[tuple[Panel, Path]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, panel in enumerate(gallery_panels(repo)):
            try:
                frame = resolve_frame(panel)
            except ValueError as exc:
                print(f"skip {panel.row} {panel.label}: {exc}")
                continue
            xyz = tmp_path / f"panel_{i:02d}.xyz"
            png = panel_dir / f"panel_{i:02d}.png"
            write_xyz(xyz, frame)
            render_panel(pymol_bin, xyz, png)
            rendered.append((panel, png))
    labelled = [(panel, png) for panel, png in rendered if panel.row in LABELLED_ROWS]
    reviewer = [(panel, png) for panel, png in rendered if panel.row not in LABELLED_ROWS]
    montage = response_figure_path(out_dir / "dataset_structures.png")
    assemble_montage(labelled, montage)
    if reviewer:
        assemble_montage(
            reviewer,
            response_figure_path(out_dir / "dataset_structures_reviewer.png"),
            pack=True,
        )
    return montage


def reassemble_gallery_from_panels(
    repo: Path | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Rebuild montage PDFs from existing PyMOL panel PNGs (no re-render)."""
    repo = repo or _REPO
    out_dir = out_dir or _DEFAULT_OUT
    panel_dir = out_dir / "structures"
    rendered: list[tuple[Panel, Path]] = []
    for i, panel in enumerate(gallery_panels(repo)):
        png = panel_dir / f"panel_{i:02d}.png"
        if png.is_file():
            rendered.append((panel, png))
    if not rendered:
        return None
    labelled = [(panel, png) for panel, png in rendered if panel.row in LABELLED_ROWS]
    reviewer = [(panel, png) for panel, png in rendered if panel.row not in LABELLED_ROWS]
    montage = response_figure_path(out_dir / "dataset_structures.png")
    assemble_montage(labelled, montage)
    if reviewer:
        assemble_montage(
            reviewer,
            response_figure_path(out_dir / "dataset_structures_reviewer.png"),
            pack=True,
        )
    return montage


def reassemble_landscape_gallery(
    repo: Path | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Wide 2x8 montage for 16:9 slides. Does not replace the letter figure."""
    repo = repo or _REPO
    out_dir = out_dir or _DEFAULT_OUT
    panel_dir = out_dir / "structures"
    rendered: list[tuple[Panel, Path]] = []
    for i, panel in enumerate(gallery_panels(repo)):
        png = panel_dir / f"panel_{i:02d}.png"
        if png.is_file():
            rendered.append((panel, png))
    labelled = [(panel, png) for panel, png in rendered if panel.row in LABELLED_ROWS]
    if not labelled:
        return None
    montage = response_figure_path(out_dir / "dataset_structures_wide.png")
    assemble_montage(labelled, montage, layout="landscape")
    return montage


def main() -> int:
    path = render_gallery()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
