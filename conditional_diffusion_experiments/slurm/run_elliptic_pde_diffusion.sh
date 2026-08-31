#!/usr/bin/env bash
# This job runs conditional diffusion and trains the elliptic PDE neural map.
# It is submitted after data generation by submit_elliptic_pde.sh.

#SBATCH --job-name=elliptic_pde_diffusion
#SBATCH --partition=batch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=elliptic_pde_diffusion_%j.out
#SBATCH --error=elliptic_pde_diffusion_%j.err

set -euo pipefail

run_dir="$SLURM_SUBMIT_DIR"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/elliptic_pde_results"

mkdir -p "$output_dir"
cd "$run_dir"

"$python_bin" -u compute_elliptic_pde_diffusion.py \
    --output-dir "$output_dir"

echo "elliptic PDE diffusion results: $output_dir"
