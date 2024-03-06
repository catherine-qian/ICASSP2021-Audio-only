#!/bin/bash
#SBATCH -J IRcls
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -p p-RTX2080
#SBATCH -A t00120220002

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate image2rev

python main_sslr.py -model 'MLP3' -epoch 30 -phaseN 10

