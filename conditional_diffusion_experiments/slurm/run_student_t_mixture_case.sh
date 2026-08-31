#!/usr/bin/env bash
# Run one task from the Student-t mixture Slurm array.

set -euo pipefail

run_dir="$SLURM_SUBMIT_DIR"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
export PATH="$pytorch_prefix/bin:$PATH"
python_bin="$pytorch_prefix/bin/python"
output_root="$run_dir/student_t_mixture_results"

# Map the six array tasks to the requested configurations.
case "$SLURM_ARRAY_TASK_ID" in
    0) dim_u=10; variance=0.01 ;;
    1) dim_u=10; variance=0.1 ;;
    2) dim_u=10; variance=0.5 ;;
    3) dim_u=15; variance=0.01 ;;
    4) dim_u=15; variance=0.1 ;;
    5) dim_u=15; variance=0.5 ;;
esac

dim_v=$((20 - dim_u))
variance_label="${variance/./p}"
output_dir="$output_root/var${variance_label}_${dim_u}u${dim_v}v"

# Preserve the original runtime settings.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$output_dir"
cd "$run_dir"

"$python_bin" -u compute_student_t_mixture.py \
    --dim-u "$dim_u" \
    --var-u-physical "$variance" \
    --var-v-physical "$variance" \
    --var-y-physical 1e-5 \
    --dm-batch 150 \
    --output-dir "$output_dir"
