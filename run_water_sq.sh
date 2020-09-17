#!/usr/bin/env bash
#$ -cwd
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -binding linear:3
#$ -q all.q    # don't fill the qlogin queue
#$ -e error_sq.txt
#$ -o output_sq.txt

cd /home/MihailBogojeski/git/schnet-tfn/equiv_dens
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ml_dft

python train_dens2.py @water_sq.txt
