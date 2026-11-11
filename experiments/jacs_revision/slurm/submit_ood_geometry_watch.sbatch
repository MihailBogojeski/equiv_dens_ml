#!/bin/bash
# Keep re-submitting ood-geom shards that die at the walltime.
# Does not touch the 3k water trees or submit tr-*.
#SBATCH --job-name=og-watch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/ood_geom_watch_%j.out

set -uo pipefail
watch_job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
submit_dir=${SLURM_SUBMIT_DIR:-$PWD}

# A previous successor chain inherited every Slurm variable on every hop until
# even mkdir/sbatch failed with E2BIG. Strip the environment using bash builtins
# before launching any external command, then export only ordinary login data.
for variable in $(compgen -e); do
  case "$variable" in
    HOME|PATH|SHELL|USER|LOGNAME|LANG|LC_*) ;;
    *) unset "$variable" ;;
  esac
done
cd "$submit_dir"
mkdir -p logs

if [[ -e results/revision/.ood_geom_watch_stop ]]; then
  echo "[$(date -Is)] stop file present; not chaining"
  exit 0
fi

successor=$(sbatch --parsable --export=NONE --dependency=afterany:"${watch_job_id}" \
  scripts/revision/submit_ood_geometry_watch.sbatch 2>&1) &&
  echo "[$(date -Is)] successor queued as ${successor}" ||
  echo "[$(date -Is)] WARNING could not queue successor: ${successor}"

ONESHOT=1 INTERVAL=60 \
  scripts/revision/watch_ood_geometry_dft.sh
