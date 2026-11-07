#!/usr/bin/env python3
"""Stack the two PyMOL panels of each frame, add titles and colour bars, and make a movie.

Input: <frames>/<stem>_rho.png and <stem>_grad.png from render_density_error.py.
Output: <frames>/composite/<stem>.png per frame, and an MP4 (ffmpeg) if --movie is given.

    .venv/bin/python experiments/jacs_revision/analysis/compose_density_movie.py \
        --frames results/revision/density_movie/traj --vmax-rho 10 --vmax-grad 20 \
        --caption-json captions.json --movie results/revision/density_movie/traj.mp4 --fps 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

GREY, MID, RED = (0.72, 0.72, 0.72), (0.93, 0.55, 0.45), (0.80, 0.05, 0.05)
BLUE, WHITE = (0.10, 0.30, 0.80), (0.92, 0.92, 0.92)
CMAPS = {"seq": LinearSegmentedColormap.from_list("grey_red", [GREY, MID, RED]),
         "div": LinearSegmentedColormap.from_list("blue_red", [BLUE, WHITE, RED])}
DEFAULT_PANELS = [
    "errrel:10:seq:%:Local density error  $100\\,|\\tilde\\rho-\\rho|/\\rho$",
    "graderrrel:20:seq:%:Local gradient error  $100\\,|\\nabla\\tilde\\rho-\\nabla\\rho|/|\\nabla\\rho|$",
]


def content_box(paths, pad: int = 20):
    """Union bounding box of non-white pixels over all images (one crop for every frame and panel)."""
    import numpy as np
    box = None
    for p in paths:
        img = plt.imread(p)[..., :3]
        mask = (img < 0.97).any(-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        b = [ys.min(), ys.max(), xs.min(), xs.max()]
        box = b if box is None else [min(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), max(box[3], b[3])]
        h, w = mask.shape
    return (max(box[0] - pad, 0), min(box[1] + pad, h), max(box[2] - pad, 0), min(box[3] + pad, w))


def compose(images: list, out: Path, caption: str, fig_w: float = 7.0, crop=None) -> None:
    """images: [(png, title, vmax, kind, unit)] stacked top to bottom."""
    imgs = [plt.imread(p) for p, *_ in images]
    if crop is not None:
        y0, y1, x0, x1 = crop
        imgs = [im[y0:y1, x0:x1] for im in imgs]
    h, w = imgs[0].shape[:2]
    n = len(images)
    fig_h = fig_w * (n * h / w) * 1.02 + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160, facecolor="white")
    top = 1 - 0.5 / fig_h
    ph = (top - 0.01) / n
    for i, (img, (_, title, vmax, kind, unit)) in enumerate(zip(imgs, images)):
        y0 = top - (i + 1) * ph
        ax = fig.add_axes([0.0, y0, 0.86, ph])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=2)
        cax = fig.add_axes([0.88, y0 + 0.2 * ph, 0.025, 0.6 * ph])
        norm = plt.Normalize(-vmax, vmax) if kind == "div" else plt.Normalize(0, vmax)
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAPS[kind], norm=norm), cax=cax)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    if caption:
        import textwrap
        fig.text(0.02, 1 - 0.25 / fig_h, "\n".join(textwrap.wrap(caption, 95)), fontsize=8.5, va="center", ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--panels", nargs="+", default=DEFAULT_PANELS,
                    help="map:vmax:kind:unit:title per panel, top to bottom (kind seq or div)")
    ap.add_argument("--caption-json", type=Path, default=None, help="{stem: caption}")
    ap.add_argument("--movie", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--ffmpeg", default="/scratch/mh7373/.pixi/bin/ffmpeg")
    opt = ap.parse_args()

    panels = [p.split(":", 4) for p in opt.panels]
    captions = json.loads(opt.caption_json.read_text()) if opt.caption_json else {}
    first = panels[0][0]
    stems = sorted(p.name[:-len(f"_{first}.png")] for p in opt.frames.glob(f"*_{first}.png"))
    outdir = opt.frames / "composite"
    crop = content_box([opt.frames / f"{stem}_{m[0]}.png" for stem in stems for m in panels])
    for k, stem in enumerate(stems):
        images = [(opt.frames / f"{stem}_{m}.png", title, float(vmax), kind, unit) for m, vmax, kind, unit, title in panels]
        compose(images, outdir / f"{k:05d}.png", captions.get(stem, ""), crop=crop)
    print(f"{len(stems)} frames -> {outdir}")
    if opt.movie:
        subprocess.run([opt.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(opt.fps), "-i", str(outdir / "%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        str(opt.movie)], check=True)
        print(f"movie -> {opt.movie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
