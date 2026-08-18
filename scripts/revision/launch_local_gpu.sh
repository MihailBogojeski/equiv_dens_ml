#!/usr/bin/env bash
# Start reviewer GPU work on the current allocation (share VRAM with gpu_burn).
# GPU 0: quick tests then PBE0 train. GPU 1: ethanol 500 ps MD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
TMUX_CONF="/exec-daemon/tmux.portal.conf"
tm() {
  if [[ -f "${TMUX_CONF}" ]]; then
    tmux -f "${TMUX_CONF}" "$@"
  else
    tmux "$@"
  fi
}
ensure() {
  local name="$1"
  tm has-session -t "=${name}" 2>/dev/null || tm new-session -d -s "${name}" -c "${ROOT}" -- "${SHELL:-bash}" -l
}

ensure rev-gpu0-quick-pbe0
ensure rev-gpu1-ethanol-md
ensure rev-wait-water

tm send-keys -t "rev-gpu0-quick-pbe0:0.0" C-c 2>/dev/null || true
tm send-keys -t "rev-gpu1-ethanol-md:0.0" C-c 2>/dev/null || true

tm send-keys -t "rev-gpu0-quick-pbe0:0.0" \
  "cd ${ROOT} && source .venv/bin/activate && export ALLOW_GL056=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 && bash scripts/revision/run_gpu_campaign.sh quick && bash scripts/revision/run_gpu_campaign.sh pbe0" \
  C-m

tm send-keys -t "rev-gpu1-ethanol-md:0.0" \
  "cd ${ROOT} && source .venv/bin/activate && export ALLOW_GL056=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 && bash scripts/revision/run_gpu_campaign.sh ethanol-md" \
  C-m

tm send-keys -t "rev-wait-water:0.0" \
  "cd ${ROOT} && source .venv/bin/activate && echo 'DFT count poller; water train is on the sbatch queue.' && while true; do date -Is; python scripts/revision/poll_dft_counts.py; sleep 600; done" \
  C-m

echo "started tmux sessions: rev-gpu0-quick-pbe0, rev-gpu1-ethanol-md, rev-wait-water"
