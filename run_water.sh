#!/usr/bin/env bash
#$ -cwd
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue
#$ -e error.txt
#$ -o output.txt

cd /home/MihailBogojeski/git/schnet-tfn/equiv_dens
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ml_dft

python train_dens2.py @water.txt
