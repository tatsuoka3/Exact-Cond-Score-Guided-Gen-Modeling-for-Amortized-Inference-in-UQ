#!/usr/bin/env bash
# Recreate the C1--C9 figure and LaTeX table from saved results.
# Run on the login node with:
#   bash slurm/run_plot_bimodal_c1_c9.sh

set -euo pipefail

slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_dir="$run_dir/bimodal_c1_c9_results"

cd "$run_dir"
"$python_bin" -u -m plot_code.plot_bimodal_c1_c9 --input-dir "$output_dir"
