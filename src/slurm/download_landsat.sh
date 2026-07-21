#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=download_landsat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=log/slurm/download_landsat.%j.out
#SBATCH --error=log/slurm/download_landsat.%j.err
  
# Your commands go here
source ~/.bashrc
acti-conda
conda activate wilds 
python3 /home/henicke/git/bigpicture/src/dataset_creation/download_landsat_geotiff.py
