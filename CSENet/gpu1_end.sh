#!/bin/bash
#SBATCH --job-name=convnext_mutil
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
source activate zxj_new
python main.py 
