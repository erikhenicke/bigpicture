#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5,gaia7
#SBATCH --job-name=eval_pc_film_om_gauss_pe_freq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/eval_pc_film_om_gauss_pe_freq.%j.out
#SBATCH --error=log/slurm/eval_pc_film_om_gauss_pe_freq.%j.err
cd /home/henicke/git/bigpicture
srun uv run --env-file .env src/results/eval_per_class.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss_pe_freq
