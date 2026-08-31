#!/usr/bin/env bash
# Rerun only the elliptic PDE plotting step with existing saved results:
#   sbatch slurm/run_elliptic_pde_plots.sh
# Existing solved plot inputs are reused. Only a missing selected field or
# missing sensor array is solved; data generation, diffusion, and MCMC are not rerun.

#SBATCH --job-name=elliptic_pde_plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=elliptic_pde_plots_%j.out
#SBATCH --error=elliptic_pde_plots_%j.err

set -euo pipefail

run_dir="$SLURM_SUBMIT_DIR"
fenics_prefix="$HOME/miniconda3/envs/fenics_env"
export PATH="$fenics_prefix/bin:$PATH"
export PKG_CONFIG_PATH="$fenics_prefix/lib/pkgconfig:$fenics_prefix/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="$fenics_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$fenics_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_bin="$fenics_prefix/bin/python"
input_dir="$run_dir/elliptic_pde_results"
output_dir="$input_dir/figures"

mkdir -p "$output_dir"
cd "$run_dir"
export MPLBACKEND=Agg

# This environment must provide FEniCS/dolfin, NumPy, SciPy, and Matplotlib.
"$python_bin" -u -m plot_code.plot_elliptic_pde \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --solve-missing

echo "elliptic PDE figures: $output_dir"
