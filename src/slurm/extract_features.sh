#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5,gaia7
#SBATCH --job-name=extract_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/extract_features.%j.out
#SBATCH --error=log/slurm/extract_features.%j.err
cd /home/henicke/git/bigpicture

# extract_features.py takes the run directory positionally; resolve the most
# recent log dir per experiment (same selection as eval_reproduce.find_run_dir).
EXPERIMENTS=(
  train_densenet_lr_baseline
  train_densenet_baseline
  train_densenet_lr_baseline_no_domain
  train_dinov3_baseline
  train_dinov3_lr_baseline
  train_dinov3_lr_baseline_no_domain
)

for experiment in "${EXPERIMENTS[@]}"; do
  run_dir=$(ls -d "log/runs/"*/"${experiment}"-* 2>/dev/null | sort | tail -n1)
  if [ -z "$run_dir" ]; then
    echo "No run directory found for ${experiment} under log/runs/" >&2
    exit 1
  fi

  echo "Using run directory for ${experiment}: $run_dir"
  start_ts=$(date -Iseconds)
  start_epoch=$(date +%s)
  echo "[$experiment] srun start: $start_ts"

  if ! srun uv run --env-file .env src/results/extract_features.py "$run_dir"; then
    end_ts=$(date -Iseconds)
    end_epoch=$(date +%s)
    elapsed=$((end_epoch - start_epoch))
    echo "[$experiment] srun end: $end_ts (failed, elapsed ${elapsed}s)" >&2
    exit 1
  fi

  end_ts=$(date -Iseconds)
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  echo "[$experiment] srun end: $end_ts (elapsed ${elapsed}s)"
done
