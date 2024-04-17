#!/bin/bash
#SBATCH -J IRcls
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -p p-V100
#SBATCH -A t00120220002

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate image2rev

inc=1
upbound=0
teA='gcc' #'gccsnr-10' #gcc
trA='gcc' #'gccsnrall'
python main_ICL.py -trA $trA -teA $teA -upbound $upbound -incremental $inc -model 'MLP3' -epoch 30 -phaseN 10

