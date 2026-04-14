#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=train_multi_deit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/train_multi_deit.%j.out
#SBATCH --error=log/slurm/train_multi_deit.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
uv run --env-file .env src/train/run_experiment.py --model_type=multi-deit --data_augmentation --image_net=both --epochs=1 --batch_size=384 --optimizer=adam --learning_rate=1e-4 --learning_rate_scheduler=plateau --learning_rate_decay=0.5 --plateau_patience=5 --frac=0.01