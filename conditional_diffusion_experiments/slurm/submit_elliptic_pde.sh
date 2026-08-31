#!/usr/bin/env bash
# Submit the elliptic PDE data, diffusion, MCMC, and plotting jobs with:
#   bash slurm/submit_elliptic_pde.sh

set -euo pipefail

slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd "$slurm_dir/.." && pwd)"

cd "$run_dir"

# Generate the shared PDE dataset first.
data_submission=$(
    sbatch --parsable \
        "$slurm_dir/run_elliptic_pde_data.sh"
)
data_job_id="${data_submission%%;*}"

# Start the independent diffusion and MCMC jobs after data generation succeeds.
diffusion_submission=$(
    sbatch --parsable \
        --dependency="afterok:$data_job_id" \
        "$slurm_dir/run_elliptic_pde_diffusion.sh"
)
diffusion_job_id="${diffusion_submission%%;*}"

mcmc_submission=$(
    sbatch --parsable \
        --dependency="afterok:$data_job_id" \
        "$slurm_dir/run_elliptic_pde_mcmc.sh"
)
mcmc_job_id="${mcmc_submission%%;*}"

# Solve the PDE for saved samples and plot after both inference jobs succeed.
plot_submission=$(
    sbatch --parsable \
        --dependency="afterok:$diffusion_job_id:$mcmc_job_id" \
        "$slurm_dir/run_elliptic_pde_plots.sh"
)
plot_job_id="${plot_submission%%;*}"

echo "submitted elliptic PDE data job $data_job_id"
echo "submitted diffusion job $diffusion_job_id after data job $data_job_id"
echo "submitted MCMC job $mcmc_job_id after data job $data_job_id"
echo "submitted plotting job $plot_job_id after diffusion and MCMC"
echo "results: $run_dir/elliptic_pde_results"
echo "figures: $run_dir/elliptic_pde_results/figures"
