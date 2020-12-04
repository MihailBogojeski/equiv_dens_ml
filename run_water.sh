#!/usr/bin/env bash
#$ -cwd
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue
#$ -e water_error.txt
#$ -o water_output.txt

cd /home/MihailBogojeski/git/schnet-tfn/equiv_dens
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ml_dft

python src/equiv_dens/train_dens2.py @water.txt
