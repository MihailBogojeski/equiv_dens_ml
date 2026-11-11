"""Watchdog must pass THEORY/RUN to sbatch via env, not bare array expansion.

Bash only treats ``VAR=value cmd`` as an environment assignment when it is a
literal prefix of a simple command. Expanding an array of ``VAR=value`` tokens
in front of ``sbatch`` makes bash try to *execute* ``THEORY=...`` as a command,
which is exactly the live failure:

    THEORY=pbe_d4_avdz: command not found

Wrapping with ``env`` restores the intended assignment semantics for the real
``sbatch`` binary.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WATCH_SCRIPTS = (
    _REPO_ROOT / "scripts" / "revision" / "watch_water_campaign.sh",
    _REPO_ROOT / "scripts" / "revision" / "water" / "watch_water_campaign.sh",
)

_BARE_ARRAY_SBATCH = re.compile(
    r'^[ \t]*"\$\{sbatch_args\[@\]\}"[ \t]+sbatch\b',
    re.MULTILINE,
)
_ENV_ARRAY_SBATCH = re.compile(
    r'^[ \t]*env[ \t]+"\$\{sbatch_args\[@\]\}"[ \t]+sbatch\b',
    re.MULTILINE,
)


@pytest.mark.parametrize("path", _WATCH_SCRIPTS, ids=lambda p: p.name + ("-themed" if "water" in p.parts[-2:] else "-entry"))
def test_training_submit_uses_env_for_sbatch_args(path: Path, implementation_text) -> None:
    text = implementation_text(path, encoding="utf-8")
    assert "sbatch_args" in text
    assert _ENV_ARRAY_SBATCH.search(text), (
        f"{path} must submit with: env \"${{sbatch_args[@]}}\" sbatch ..."
    )
    assert not _BARE_ARRAY_SBATCH.search(text), (
        f"{path} still expands sbatch_args as a bare command prefix"
    )


def test_env_array_prefix_sets_variables_for_external_command() -> None:
    """Behavioral lock for the bash pitfall the watchdog hit in production."""
    script = r"""
set -euo pipefail
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
cat >"$tmpdir/printenv.sh" <<'EOF'
#!/bin/bash
printf 'THEORY=%s;RUN=%s;INIT_SEED=%s\n' "${THEORY-}" "${RUN-}" "${INIT_SEED-}"
EOF
chmod +x "$tmpdir/printenv.sh"
export PATH="$tmpdir:$PATH"
# Shadow real binaries: our printenv.sh is named sbatch via symlink.
ln -sf "$tmpdir/printenv.sh" "$tmpdir/sbatch"

sbatch_args=(THEORY=pbe_d4_avdz RUN=cutoff_4 INIT_SEED=7)

# Bare expansion must fail the way production failed.
if "${sbatch_args[@]}" sbatch >/tmp/bare_out 2>/tmp/bare_err; then
  echo "bare expansion unexpectedly succeeded" >&2
  exit 1
fi
grep -q 'THEORY=pbe_d4_avdz: command not found' /tmp/bare_err

out=$(env "${sbatch_args[@]}" sbatch)
test "$out" = "THEORY=pbe_d4_avdz;RUN=cutoff_4;INIT_SEED=7"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
