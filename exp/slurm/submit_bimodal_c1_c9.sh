#!/usr/bin/env bash
# Run this file on the login node with:
#   bash slurm/submit_bimodal_c1_c9.sh

set -euo pipefail

# Locate the experiment directory from this script's location.
slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/bimodal_c1_c9_results"

mkdir -p "$output_dir"
cd "$run_dir"

# Run c1 through c9 as nine tasks in one Slurm array.
compute_submission=$(
    sbatch --parsable \
        --job-name="bimodal_c1_c9_compute" \
        --partition=batch \
        --array=1-9 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=02:00:00 \
        --chdir="$run_dir" \
        --output="$run_dir/bimodal_c1_c9_%A_%a.out" \
        --error="$run_dir/bimodal_c1_c9_%A_%a.err" \
        --wrap="$python_bin -u $run_dir/compute_bimodal_c1_c9.py --config-index \$SLURM_ARRAY_TASK_ID --device cpu --output-dir $output_dir"
)
compute_job_id="${compute_submission%%;*}"

# Create the figure and LaTeX table after all nine tasks succeed.
output_submission=$(
    sbatch --parsable \
        --job-name="bimodal_c1_c9_output" \
        --partition=batch \
        --cpus-per-task=1 \
        --mem=4G \
        --time=02:00:00 \
        --chdir="$run_dir" \
        --output="$run_dir/bimodal_c1_c9_output_%j.out" \
        --error="$run_dir/bimodal_c1_c9_output_%j.err" \
        --dependency="afterok:$compute_job_id" \
        --wrap="$python_bin -u -m plot_code.plot_bimodal_c1_c9 --input-dir $output_dir"
)
output_job_id="${output_submission%%;*}"

# Print the submitted job numbers and final output locations.
echo "submitted c1-c9 array job $compute_job_id"
echo "submitted figure/table job $output_job_id"
echo "results: $output_dir"
echo "figures: $output_dir/figures/fig1_bimodal.png and fig1_bimodal.pdf"
echo "latex table: $output_dir/summary_table.tex"
