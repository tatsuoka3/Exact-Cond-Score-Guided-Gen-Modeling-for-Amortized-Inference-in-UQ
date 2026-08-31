#!/usr/bin/env bash
# This job trains the elliptic PDE surrogate and runs MCMC for both test cases.
# It is submitted after data generation by submit_elliptic_pde.sh.

#SBATCH --job-name=elliptic_pde_mcmc
#SBATCH --partition=batch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=elliptic_pde_mcmc_%j.out
#SBATCH --error=elliptic_pde_mcmc_%j.err

set -euo pipefail

run_dir="$SLURM_SUBMIT_DIR"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/elliptic_pde_results"

mkdir -p "$output_dir"
cd "$run_dir"

# The PyTorch environment must also provide emcee.
"$python_bin" -u compute_elliptic_pde_mcmc.py \
    --output-dir "$output_dir"

echo "elliptic PDE MCMC results: $output_dir/elliptic_pde_mcmc"
echo "samples: $output_dir/MCMC_samples_testcase1.npy and MCMC_samples_testcase2.npy"
