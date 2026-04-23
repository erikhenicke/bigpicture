#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5,nyx
#SBATCH --job-name=train_multi_densenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=slurm/train_multi_densenet.%j.out
#SBATCH --error=slurm/train_multi_densenet.%j.err
  
# Your commands go here
cd /home/henicke/git/bigpicture
srun uv run --env-file .env src/train/run_experiment.py --model_type=multi-densenet --data_augmentation --image_net=both --epochs=50 --batch_size=64 --frac=1 --optimizer=adam --learning_rate=1e-4 --learning_rate_scheduler=step --learning_rate_decay=0.96