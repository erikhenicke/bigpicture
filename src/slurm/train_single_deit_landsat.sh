#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=train_single_deit_landsat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/train_single_deit_landsat.%j.out
#SBATCH --error=slurm/train_single_deit_landsat.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=single-deit-landsat --data_augmentation --image_net=both --epochs=50 --batch_size=384 --frac=1.0 --optimizer=adam --learning_rate=1e-4 --learning_rate_scheduler=plateau --learning_rate_decay=0.5 --plateau_patience=5

