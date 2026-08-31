#!/usr/bin/env bash
# Submit the 10/10 and 15/5 Student-t score-network projection-KL jobs with:
#   bash slurm/submit_student_t_score_network_kl.sh

set -euo pipefail

slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"
pytorch_prefix="$HOME/miniconda3/envs/pytorch_gpu_env"
python_bin="$pytorch_prefix/bin/python"
compute_script="$run_dir/compute_student_t_score_network_kl.py"
output_root="$run_dir/student_t_score_network_results"

mkdir -p "$output_root"
cd "$run_dir"

for dim_u in 10 15; do
    dim_v=$((20 - dim_u))
    output_dir="$output_root/dim${dim_u}_${dim_v}"

    submission=$(
        sbatch --parsable \
            --job-name="student_t_score_${dim_u}_${dim_v}" \
            --partition=batch-gpu \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gres=gpu:v100:1 \
            --mem=8G \
            --time=02:00:00 \
            --chdir="$run_dir" \
            --output="$run_dir/student_t_score_${dim_u}_${dim_v}_%j.out" \
            --error="$run_dir/student_t_score_${dim_u}_${dim_v}_%j.err" \
            --wrap="export PATH='$pytorch_prefix/bin':\$PATH OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'; '$python_bin' -u '$compute_script' --dim-u '$dim_u' --device cuda --output-dir '$output_dir'"
    )

    job_id="${submission%%;*}"
    echo "submitted ${dim_u}/${dim_v} score-network job $job_id"
done

echo "results: $run_dir/student_t_score_network_results"
