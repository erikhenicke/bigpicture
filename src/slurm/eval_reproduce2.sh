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
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_lr_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_lr_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_lr_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_lr_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_stacked_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_stacked_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_stacked_baseline

# ---------------- multsim.yaml (21 runs with 3 seeds) ----------------
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name geoprior_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_relu
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss_pe
