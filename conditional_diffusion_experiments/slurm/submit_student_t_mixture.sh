#!/usr/bin/env bash
# Submit the six Student-t mixture runs and their final plotting job with:
#   bash slurm/submit_student_t_mixture.sh

set -euo pipefail

slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/student_t_mixture_results"

mkdir -p "$output_dir"
cd "$run_dir"

# Run all six variance/dimension configurations as one GPU array.
compute_submission=$(
    sbatch --parsable \
        --job-name="student_t_mixture" \
        --partition=batch-gpu \
        --array=0-5 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --gres=gpu:v100:1 \
        --mem=8G \
        --time=02:00:00 \
        --chdir="$run_dir" \
        --output="$run_dir/student_t_mixture_%A_%a.out" \
        --error="$run_dir/student_t_mixture_%A_%a.err" \
        "$run_dir/slurm/run_student_t_mixture_case.sh"
)
compute_job_id="${compute_submission%%;*}"

# Plot only after all six array tasks finish successfully.
plot_submission=$(
    sbatch --parsable \
        --job-name="student_t_mixture_plot" \
        --partition=batch \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=8G \
        --time=02:00:00 \
        --chdir="$run_dir" \
        --output="$run_dir/student_t_mixture_plot_%j.out" \
        --error="$run_dir/student_t_mixture_plot_%j.err" \
        --dependency="afterok:$compute_job_id" \
        --wrap="export MPLBACKEND=Agg; $python_bin -u -m plot_code.plot_student_t_mixture --input-dir $output_dir --output-dir $output_dir/figures"
)
plot_job_id="${plot_submission%%;*}"

echo "submitted Student-t mixture array job $compute_job_id"
echo "submitted plotting job $plot_job_id"
echo "results: $output_dir"
echo "figures: $output_dir/figures"
