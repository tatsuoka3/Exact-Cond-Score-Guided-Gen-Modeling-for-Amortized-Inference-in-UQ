#!/usr/bin/env bash
# Submit this single job from the login node with:
#   sbatch slurm/run_bimodal_time_experiment.sh

# Slurm resources for the computation and plotting steps.
#SBATCH --job-name=bimodal_time
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=bimodal_time_%j.out
#SBATCH --error=bimodal_time_%j.err

set -euo pipefail

# Slurm records the directory from which sbatch was called.
run_dir="$SLURM_SUBMIT_DIR"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/time_results"

mkdir -p "$output_dir"
cd "$run_dir"

# Compute and save the time-discretization results.
"$python_bin" -u compute_bimodal_time_experiment.py \
    --device cpu \
    --output-dir "$output_dir"

# Plot the saved results in the same job.
"$python_bin" -u -m plot_code.plot_bimodal_time_experiment \
    --input-file "$output_dir/time_experiment_results.npz"

# Print the final output locations in the job log.
echo "results: $output_dir/time_experiment_results.npz"
echo "figures: $output_dir/figures/fig2_time.png and fig2_time.pdf"
