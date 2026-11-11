#!/usr/bin/env python3
"""Two-panel PyMOL rendering of where DenSNet's density is wrong.

Both panels show the same DFT isodensity surface (the "electron cloud") of one
structure, with sticks for the nuclei. The surface is grey where DenSNet agrees
with DFT and turns red where it does not:

  top    : local density error      100 |rho_DenSNet - rho_DFT| / rho_DFT
  bottom : local gradient error     100 |grad rho_DenSNet - grad rho_DFT| / |grad rho_DFT|

On an isodensity surface rho_DFT is constant, so the top panel is also the
absolute error up to a constant. Colour scales are fixed (--vmax-rho,
--vmax-grad) so frames of a movie are comparable. Inputs are the cubes written
by density_error_cubes.py.

Run with PyMOL's own Python (open-source PyMOL, headless):
    pymol -cq experiments/jacs_revision/analysis/render_density_error.py -- \
        --cubes results/revision/density_movie/test --tag ood_modes_ethanethiol_pbe_d4_avdz_0000 \
        --out results/revision/density_movie/test/frame_0000.png
Frames are then stacked and labelled by compose_density_movie.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pymol import cmd

GREY, RED = [0.72, 0.72, 0.72], [0.80, 0.05, 0.05]
MID = [0.93, 0.55, 0.45]  # salmon, so small errors are visible before full red
BLUE, WHITE = [0.10, 0.30, 0.80], [0.92, 0.92, 0.92]


def render_panel(ref: Path, err: Path, level: float, vmax: float, view, out: Path, width: int, height: int,
                 transparency: float, kind: str = "seq", scale: float = 1.0) -> None:
    """scale: the coloured quantity is scale * (map); the ramp is rescaled (and flipped if scale < 0)."""
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadow", 0)
    cmd.set("specular", 0.25)
    cmd.set("ambient", 0.45)
    cmd.set("orthoscopic", 1)
    cmd.load(str(ref), "rho")          # also loads the atoms as object "rho_molecule"? (cube -> map)
    cmd.load(str(err), "errmap")
    cmd.isosurface("cloud", "rho", level)
    v = vmax / max(abs(scale), 1e-3)
    if kind == "div":  # signed quantity: blue (negative) - grey (zero) - red (positive)
        cmd.ramp_new("ramp", "errmap", [-v, 0.0, v], [BLUE, WHITE, RED] if scale >= 0 else [RED, WHITE, BLUE])
    else:              # error magnitude: grey (none) - salmon - red
        cmd.ramp_new("ramp", "errmap", [0.0, 0.5 * v, v], [GREY, MID, RED])
    cmd.color("ramp", "cloud")
    cmd.disable("ramp")
    cmd.set("transparency", transparency, "cloud")
    cmd.set("transparency_mode", 3)  # front layer only: see the nuclei, not the far side of the cloud
    cmd.disable("rho")
    cmd.disable("errmap")
    # nuclei: cube headers carry the atoms; PyMOL creates no molecule from a cube, so use the xyz
    xyz = ref.with_suffix(".xyz")
    if xyz.is_file():
        cmd.load(str(xyz), "mol")
        # ball-and-stick, CPK colours
        cmd.show_as("sticks", "mol")
        cmd.show("spheres", "mol")
        cmd.set("stick_radius", 0.14)
        cmd.set("sphere_scale", 0.24)
        cmd.set("sphere_scale", 0.16, "mol and elem H")
        cmd.set("stick_color", "grey60")
        cmd.set_color("cpk_c", [0.35, 0.35, 0.35])
        cmd.set_color("cpk_s", [0.95, 0.80, 0.15])
        cmd.color("cpk_c", "mol and elem C")
        cmd.color("white", "mol and elem H")
        cmd.color("cpk_s", "mol and elem S")
        cmd.color("red", "mol and elem O")
        cmd.color("blue", "mol and elem N")
        cmd.set("stick_color", "default")
    if view is not None:
        cmd.set_view(view)
    else:
        cmd.orient("mol" if xyz.is_file() else "cloud")
        cmd.zoom("mol" if xyz.is_file() else "cloud", 2.0, complete=1)
    cmd.png(str(out), width=width, height=height, dpi=150, ray=1)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output stem; writes <stem>_<map>.png per panel")
    ap.add_argument("--panels", nargs="+", default=["errrel:10:seq", "graderrrel:20:seq"],
                    help="map:vmax:kind per panel; map is the cube suffix, kind seq (grey-red) or div (blue-grey-red)")
    ap.add_argument("--level", type=float, default=0.03, help="isodensity level (e/bohr^3)")
    ap.add_argument("--view", type=Path, default=None, help="file with a PyMOL view (18 floats) for a fixed camera")
    ap.add_argument("--save-view", type=Path, default=None)
    ap.add_argument("--fit-all", action="store_true", help="frame all *_ref.xyz structures in --cubes")
    ap.add_argument("--map-base", type=Path, default=None,
                    help="path prefix of the colour maps (<prefix>_<map>.cube); default <cubes>/<tag>")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplies every panel not marked ':static' (vibration phase, from mode_vibration_frames.py)")
    ap.add_argument("--fit-buffer", type=float, default=2.5, help="Angstrom added around the atoms for --fit-all")
    ap.add_argument("--transparency", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=800)
    opt = ap.parse_args(argv)

    view = [float(x) for x in opt.view.read_text().split()] if opt.view else None
    if opt.fit_all:
        # one camera that frames every structure of the movie (orientation from the first)
        cmd.reinitialize()
        files = sorted(opt.cubes.glob("*_ref.xyz"))
        for i, f in enumerate(files):
            cmd.load(str(f), "all", state=i + 1)
        cmd.orient("all", state=1)
        cmd.zoom("all", opt.fit_buffer, state=0)
        view = list(cmd.get_view())
    base = opt.cubes / opt.tag
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    maps = opt.map_base or base
    for i, panel in enumerate(opt.panels):
        name, vmax, kind, *flags = panel.split(":")
        scale = 1.0 if "static" in flags else opt.scale
        render_panel(Path(f"{base}_ref.cube"), Path(f"{maps}_{name}.cube"), opt.level, float(vmax), view,
                     Path(f"{opt.out}_{name}.png"), opt.width, opt.height, opt.transparency, kind, scale)
        if i == 0:
            if opt.save_view:
                opt.save_view.write_text(" ".join(f"{v:.6f}" for v in cmd.get_view()) + "\n")
            view = view or list(cmd.get_view())
    return 0


main()
