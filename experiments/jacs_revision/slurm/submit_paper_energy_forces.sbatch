#!/bin/bash
# Measured E/F MAE of the paper checkpoints (Table S2 / Fig. 4e) on their test
# sets, one structure at a time. CPU only; the three models run side by side.
#SBATCH --job-name=paper-ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --output=logs/paper_ef_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 CUDA_VISIBLE_DEVICES="" WANDB_MODE=offline
out=results/revision/paper_energy_forces
score() {  # name run test train [args-file]
  ./.venv/bin/python -B experiments/jacs_revision/analysis/score_paper_energy_forces.py --device cpu \
    --run "$2" --test-npy "$3" --train-npy "$4" ${5:+--args-file "$5"} --out "$out/$1" > "logs/paper_ef_${SLURM_JOB_ID}_$1.log" 2>&1
}
score ethanethiol_delta paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4 \
  datasets/ethanethiol_rand-1000.npy datasets/ethanethiol_combo_Aidx-1000.npy config/md/nn/ethanethiol.txt &
score ethanethiol_direct paper/models/ethanethiol/2025-06-08_cnRa8AqQ \
  datasets/ethanethiol_rand-1000.npy datasets/ethanethiol_combo_Aidx-1000.npy &
score resorcinol_delta paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1 \
  datasets/resorcinol_augccpvdz_test.npy datasets/resorcinol_combo_kmeansidx-1000_train.npy config/md/nn/resorcinol.txt &
wait
tail -n 2 logs/paper_ef_${SLURM_JOB_ID}_*.log
