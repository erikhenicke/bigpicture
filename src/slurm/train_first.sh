#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=train_deit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=train_deit.%j.out
#SBATCH --error=train_deit.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds/bin/python /home/henicke/git/bigpicture/src/train/train_first.py --model=multi --epochs=1 --batch_size=32 --frac=0.01
