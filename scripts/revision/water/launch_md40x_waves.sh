#!/usr/bin/env bash
# Compatibility entry point. Implementation: experiments/jacs_revision/water/launch_md40x_waves.sh
revision_repo_root() {
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pyproject.toml" ]]; then printf '%s\n' "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  echo "could not find repository root" >&2
  return 1
}
root="$(revision_repo_root)" || exit $?
exec bash "$root/experiments/jacs_revision/water/launch_md40x_waves.sh" "$@"
