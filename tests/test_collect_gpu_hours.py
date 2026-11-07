import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts/revision/analysis/collect_gpu_hours.py"
_SPEC = importlib.util.spec_from_file_location("collect_gpu_hours", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_elapsed_to_hours = _MOD._elapsed_to_hours
_row_and_stage = _MOD._row_and_stage


def test_elapsed_parses_days_and_hms():
    assert _elapsed_to_hours("01:30:00") == 1.5
    assert abs(_elapsed_to_hours("1-02:00:00") - 26.0) < 1e-9
    assert _elapsed_to_hours("Elapsed") is None


def test_row_maps_water_and_energy_stage():
    row, stage = _row_and_stage("tre-water-wb97mv_def2tzvpd")
    assert row.startswith("Water")
    assert stage == "energy"
    row, stage = _row_and_stage("tr-malonaldehyde-pbe_d4_avdz")
    assert row == "Malonaldehyde, PBE-D4"
    assert stage == "density"
