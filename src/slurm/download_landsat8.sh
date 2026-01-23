#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=download_landsat8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=download_landsat8.%j.out
#SBATCH --error=download_landsat8.%j.err
  
# Your commands go here
source ~/.bashrc
acti-conda
conda activate wilds 
python3 /home/henicke/git/bigpicture/src/dataset_creation/download_landsat8.py