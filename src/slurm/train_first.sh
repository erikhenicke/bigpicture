#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=download_landsat8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=download_landsat8.%j.out
#SBATCH --error=download_landsat8.%j.err
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds/bin/python /home/henicke/git/bigpicture/src/train/train_first.py
