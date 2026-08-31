#!/usr/bin/env bash
# Submit every experiment and its plotting jobs with:
#   bash slurm/submit_all.sh

set -euo pipefail

slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"

cd "$run_dir"

# Submit the bimodal C1--C9 array and dependent output job.
bash "$slurm_dir/submit_bimodal_c1_c9.sh"

# Submit the single bimodal time computation/plotting job.
time_submission=$(sbatch --parsable "$slurm_dir/run_bimodal_time_experiment.sh")
time_job_id="${time_submission%%;*}"
echo "submitted bimodal time job $time_job_id"

# Submit the six Student-t mixture tasks and dependent plotting job.
bash "$slurm_dir/submit_student_t_mixture.sh"

# Submit elliptic PDE data generation followed by diffusion and MCMC.
bash "$slurm_dir/submit_elliptic_pde.sh"
