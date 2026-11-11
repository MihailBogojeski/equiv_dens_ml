"""The watchdog snapshot must still land in the checkout.

``submit_water_watchdog.sbatch`` copies ``watch_water_campaign.sh`` to
``logs/watchdog/campaign_<jobid>.sh`` so edits to the repository copy cannot
shift a running shell's byte offset. The themed-folder move put the original
three directories below the root (``scripts/revision/water/``), and the script
counted ``../../..`` to get home. From the snapshot that is one directory
*above* the checkout, which is how a live hop started walking sibling projects
instead of ``datasets/revision/shards``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CAMPAIGN = _REPO_ROOT / "scripts" / "revision" / "watch_water_campaign.sh"


def _startup_through_cd(src: str) -> str:
    """Keep the campaign script through its startup ``cd``, then print ``pwd``."""
    out: list[str] = []
    seen_cd = False
    for line in src.splitlines(True):
        out.append(line)
        stripped = line.strip()
        if stripped.startswith("cd ") and "dirname" not in stripped.split("#", 1)[0]:
            # ``cd "$(revision_repo_root)"`` or any other non-comment cd
            if "dirname" not in stripped:
                seen_cd = True
                break
        if stripped.startswith("cd ") and "dirname" in stripped:
            seen_cd = True
            break
    if not seen_cd:
        raise AssertionError("watch_water_campaign.sh has no startup cd")
    out.append("pwd\n")
    return "".join(out)


def _pwd_after_startup(script_path: Path) -> Path:
    env = os.environ.copy()
    env.pop("SLURM_SUBMIT_DIR", None)
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip()).resolve()


def test_snapshot_copy_cds_to_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "equiv_dens_ml"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'equiv_dens_ml'\n")
    snapshot = checkout / "logs" / "watchdog" / "campaign.sh"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(_startup_through_cd(_CAMPAIGN.read_text()))
    snapshot.chmod(0o755)
    assert _pwd_after_startup(snapshot) == checkout.resolve()


def test_themed_copy_still_cds_to_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "equiv_dens_ml"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'equiv_dens_ml'\n")
    themed = checkout / "scripts" / "revision" / "water" / "watch_water_campaign.sh"
    themed.parent.mkdir(parents=True)
    themed.write_text(_startup_through_cd(_CAMPAIGN.read_text()))
    themed.chmod(0o755)
    assert _pwd_after_startup(themed) == checkout.resolve()
