#!/usr/bin/env bash
# This job generates the elliptic PDE training data and two test cases.
# Submit the complete workflow with:
#   bash slurm/submit_elliptic_pde.sh

#SBATCH --job-name=elliptic_pde_data
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=elliptic_pde_data_%j.out
#SBATCH --error=elliptic_pde_data_%j.err

set -euo pipefail

run_dir="$SLURM_SUBMIT_DIR"
fenics_prefix="$HOME/miniconda3/envs/fenics_env"
export PATH="$fenics_prefix/bin:$PATH"
export PKG_CONFIG_PATH="$fenics_prefix/lib/pkgconfig:$fenics_prefix/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="$fenics_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$fenics_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_bin="$fenics_prefix/bin/python"
output_dir="$run_dir/elliptic_pde_results"

mkdir -p "$output_dir"
cd "$run_dir"

# This Python environment must provide FEniCS/dolfin, NumPy, and Matplotlib.
"$python_bin" -u generate_elliptic_pde_data.py \
    --output-dir "$output_dir"

echo "elliptic PDE data: $output_dir/data"
