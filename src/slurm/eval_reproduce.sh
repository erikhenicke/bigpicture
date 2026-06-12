#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia7
#SBATCH --job-name=reproduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/reproduce.%j.out
#SBATCH --error=log/slurm/reproduce.%j.err
cd /home/henicke/git/bigpicture

# ---------------- baselines.yaml (15 runs with 3 seeds) ----------------
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name efficientformer_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name efficientnet_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name deit_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_image_enc_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_le_enc_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_le_enc_baseline_no_domain

# ---------------- multsim.yaml (21 runs with 3 seeds) ----------------
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name geoprior
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat_detach
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_lr
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_hr
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_both
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_detach
