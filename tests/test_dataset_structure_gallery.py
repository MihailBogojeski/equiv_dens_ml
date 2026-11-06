"""Select representative frames for the response-letter structure gallery."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

from render_dataset_structures import (
    MALON_TRAIN_DELTA_MAX,
    Frame,
    Panel,
    gallery_panels,
    heavy_min_distance,
    malon_region,
    n_water,
    parse_delta,
    resolve_frame,
    select_first_clean,
    select_first_with_n,
    select_mid_with_n,
    select_nearest_delta,
    write_overlay_pml,
    write_pml,
    write_xyz,
)


def _water(n: int, comment: str) -> Frame:
    symbols = (["O", "H", "H"] * n)
    coords = [[float(i), 0.0, 0.0] for i in range(3 * n)]
    return Frame(symbols=symbols, coords=coords, comment=comment)


def test_n_water_counts_oxygens():
    assert n_water(["O", "H", "H", "O", "H", "H"]) == 2


def test_heavy_min_distance_ignores_hydrogens():
    frame = Frame(
        ["C", "H", "C"],
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.5, 0.0, 0.0]],
        "clashy",
    )
    assert heavy_min_distance(frame) == pytest.approx(1.5)


def test_select_first_clean_skips_fused_frames():
    fused = Frame(
        ["C", "C", "O", "O"],
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
        "fused",
    )
    ok = Frame(
        ["C", "C", "O", "O"],
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
        "ok",
    )
    chosen = select_first_clean([fused, fused, ok])
    assert chosen.comment == "ok"


def test_select_first_with_n_skips_smaller_clusters():
    frames = [_water(2, "n=2"), _water(8, "n=8 a"), _water(8, "n=8 b")]
    chosen = select_first_with_n(frames, 8)
    assert chosen.comment == "n=8 a"
    assert n_water(chosen.symbols) == 8


def test_select_mid_with_n_picks_the_middle_match():
    frames = [
        _water(2, "n=2"),
        _water(8, "n=8 a"),
        _water(8, "n=8 b"),
        _water(8, "n=8 c"),
        _water(6, "n=6"),
    ]
    chosen = select_mid_with_n(frames, 8)
    assert chosen.comment == "n=8 b"


def test_parse_delta_from_malon_comment():
    assert parse_delta("malonaldehyde scan delta=-0.007") == pytest.approx(-0.007)


def test_select_nearest_delta_picks_the_crossing():
    frames = [
        Frame(["O"], [[0.0, 0.0, 0.0]], "malonaldehyde scan delta=-0.76"),
        Frame(["O"], [[0.0, 0.0, 0.0]], "malonaldehyde scan delta=-0.01"),
        Frame(["O"], [[0.0, 0.0, 0.0]], "malonaldehyde scan delta=0.58"),
    ]
    chosen = select_nearest_delta(frames, 0.0)
    assert chosen.comment.endswith("-0.01")


def test_select_nearest_delta_skips_comments_without_delta():
    frames = [
        Frame(["O"], [[0.0, 0.0, 0.0]], "acac scan-jitter"),
        Frame(["O"], [[0.0, 0.0, 0.0]], "acac scan delta=0.02"),
        Frame(["O"], [[0.0, 0.0, 0.0]], "acac scan-jitter"),
    ]
    chosen = select_nearest_delta(frames, 0.0)
    assert "0.02" in chosen.comment


def test_write_xyz_roundtrip(tmp_path: Path):
    frame = _water(2, "water n=2 dimer")
    path = tmp_path / "dimer.xyz"
    write_xyz(path, frame)
    text = path.read_text()
    assert text.startswith("6\nwater n=2 dimer\n")
    assert "O " in text


def test_overlay_pml_uses_gallery_stick_style(tmp_path: Path):
    dft = tmp_path / "dft.xyz"
    ml = tmp_path / "ml.xyz"
    png = tmp_path / "overlay.png"
    write_xyz(dft, _water(1, "dft"))
    write_xyz(ml, _water(1, "ml"))
    single = write_pml(dft, png)
    overlay = write_overlay_pml(dft, ml, png)
    for snippet in (
        "set stick_radius, 0.18",
        "set sphere_scale, 0.16",
        "set valence, 0",
        "bg_color white",
        "turn x, 22",
        "turn y, 28",
    ):
        assert snippet in single
        assert snippet in overlay
    assert "load" in overlay and "dft" in overlay and "ml" in overlay
    assert "util.cbaw dft" in overlay
    assert "color tv_blue, ml" in overlay


def test_write_aligned_xyz_superimposes_a_rotated_copy(tmp_path: Path):
    from render_dataset_structures import write_aligned_xyz, read_xyz_coords
    import numpy as np

    ref = tmp_path / "ref.xyz"
    write_xyz(ref, _water(1, "ref"))
    symbols, coords = read_xyz_coords(ref)
    moved = coords + np.array([1.5, -0.4, 0.2])
    mobile = tmp_path / "mobile.xyz"
    write_xyz(mobile, Frame(symbols=symbols, coords=moved.tolist(), comment="moved"))
    aligned = write_aligned_xyz(mobile, ref, tmp_path / "aligned.xyz")
    _, got = read_xyz_coords(aligned)
    assert np.allclose(got, coords, atol=1e-8)


@pytest.mark.local_data
def test_gallery_uses_thermal_and_rackers_frames_not_only_minima():
    panels = gallery_panels(_REPO)
    rows = [p.row for p in panels]
    assert rows.count("Training") == 3
    assert rows.count("Size hold-out") == 3
    assert "Ice" in rows
    assert "Droplets" in rows
    assert "Rackers" in rows
    assert rows.count("Malonaldehyde") == 3
    train_sources = {p.source.name for p in panels if p.row == "Training"}
    assert train_sources == {"train.xyz"}
    assert any(p.kind == "mid_n" for p in panels if p.row == "Training")
    rackers = [p for p in panels if p.row == "Rackers"]
    assert [p.target for p in rackers] == [8, 15, 30]
    for panel in panels:
        assert panel.source.is_file(), panel.source
    extra = {p.row for p in panels}
    assert {"MeOH–water", "Paracetamol"} <= extra
    assert "Crambin frags" not in extra
    from render_dataset_structures import optional_reviewer_panels

    assert extra >= {p.row for p in optional_reviewer_panels(_REPO)}


def test_optional_reviewer_panels_only_when_xyz_exist(tmp_path):
    from render_dataset_structures import optional_reviewer_panels

    assert optional_reviewer_panels(tmp_path) == []
    nma = tmp_path / "datasets" / "revision" / "nma_dimer"
    nma.mkdir(parents=True)
    (nma / "train.xyz").write_text(
        "4\nnma\nC 0 0 0\nC 1.5 0 0\nO 2.7 0 0\nN -1.4 0 0\n"
    )
    meoh = tmp_path / "datasets" / "revision" / "meoh_water"
    meoh.mkdir(parents=True)
    (meoh / "train.xyz").write_text("3\nmeoh\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    acac = tmp_path / "datasets" / "revision" / "acac"
    acac.mkdir(parents=True)
    (acac / "train.xyz").write_text("3\nacac train delta=-0.80\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    (acac / "ood_proton_transfer.xyz").write_text(
        "3\nacac scan delta=0.00\nC 0 0 0\nO 1 0 0\nH 0 1 0\n"
    )
    rows = [p.row for p in optional_reviewer_panels(tmp_path)]
    assert "NMA dimer" in rows
    assert "MeOH–water" in rows
    assert "AcAc" in rows
    par = tmp_path / "datasets" / "revision" / "paracetamol"
    par.mkdir()
    (par / "train.xyz").write_text("3\npar\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    rows = [p.row for p in optional_reviewer_panels(tmp_path)]
    assert "Paracetamol" in rows


def test_landscape_montage_is_wider_than_tall(tmp_path: Path):
    import matplotlib.pyplot as plt

    from render_dataset_structures import assemble_montage

    rows = (
        ["Training"] * 3
        + ["Size hold-out"] * 3
        + ["Ice"] * 2
        + ["Droplets"] * 2
        + ["Rackers"] * 3
        + ["Malonaldehyde"] * 3
    )
    panels = []
    for i, row in enumerate(rows):
        png = tmp_path / f"p{i:02d}.png"
        fig, ax = plt.subplots(figsize=(0.4, 0.4))
        ax.axis("off")
        fig.savefig(png, dpi=40)
        plt.close(fig)
        panels.append((Panel(row, f"$n={i}$", png), png))
    out = tmp_path / "wide.pdf"
    assemble_montage(panels, out, layout="landscape")
    assert out.is_file()
    text = out.read_bytes()
    box = None
    for marker in (b"/MediaBox", b"/BBox"):
        idx = text.find(marker)
        if idx < 0:
            continue
        start = text.find(b"[", idx)
        end = text.find(b"]", start)
        nums = [float(x) for x in text[start + 1 : end].split()]
        if len(nums) == 4:
            box = nums
            break
    assert box is not None
    width = box[2] - box[0]
    height = box[3] - box[1]
    assert width / height > 1.6


def test_malon_region_splits_on_the_training_cutoff():
    assert MALON_TRAIN_DELTA_MAX == pytest.approx(-0.40)
    assert malon_region(-0.80) == "training"
    assert malon_region(-0.40) == "training"
    assert malon_region(-0.39) == "held-out"
    assert malon_region(0.0) == "held-out"
    assert malon_region(0.58) == "held-out"


def test_malon_error_points_join_on_source_index_and_split_regions():
    from plot_response_letter_figures import malon_error_points

    frames = [
        {"index": 0, "delta_pt": -0.80},
        {"index": 1, "delta_pt": 0.02},
        {"index": 2, "delta_pt": -0.50},
    ]
    records = [
        {"index": 99, "source_index": 0, "afe": 0.003},
        {"index": 99, "source_index": 1, "afe": 0.006},
    ]
    points = malon_error_points(frames, records)
    assert len(points) == 2
    by_delta = {round(p["delta"], 2): p for p in points}
    assert by_delta[-0.8]["region"] == "training"
    assert by_delta[0.02]["region"] == "held-out"
    assert by_delta[0.02]["afe"] == pytest.approx(0.006)


def test_resolve_frame_ice_and_malon_scan():
    ice = resolve_frame(next(p for p in gallery_panels(_REPO) if p.row == "Ice"))
    assert n_water(ice.symbols) == 12
    assert "ice" in ice.comment.lower()
    malon_ts = resolve_frame(
        next(p for p in gallery_panels(_REPO) if p.kind == "delta" and p.target == 0.0)
    )
    assert abs(parse_delta(malon_ts.comment)) < 0.05
