#!/bin/bash
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t folders < <(find "$SLURM_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#folders[@]} -eq 0 ]]; then
    echo "No subfolders found in $SLURM_DIR" >&2
    exit 1
fi

echo "Available folders:"
for i in "${!folders[@]}"; do
    echo "  [$i] $(basename "${folders[$i]}")"
done

read -rp "Select folder index: " idx

if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx >= ${#folders[@]} )); then
    echo "Invalid selection" >&2
    exit 1
fi

selected="${folders[$idx]}"
scripts=("$selected"/*.sh)

if [[ ! -e "${scripts[0]}" ]]; then
    echo "No .sh files found in $(basename "$selected")" >&2
    exit 1
fi

echo "Submitting jobs from $(basename "$selected"):"
for script in "${scripts[@]}"; do
    job_id=$(sbatch "$script" | awk '{print $NF}')
    echo "  submitted $(basename "$script") → job $job_id"
done
