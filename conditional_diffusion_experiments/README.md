## Conditional diffusion experiments

This directory contains the bimodal C1--C9 experiment, bimodal
time-discretization experiment, six-run Student-t mixture experiment,
Student-t score-network comparison, and elliptic PDE example. The diffusion
experiments share `shared_experiment_utils.py`.

    conditional_diffusion_experiments/
    ├── shared_experiment_utils.py
    ├── bimodal_experiment.py
    ├── compute_bimodal_c1_c9.py
    ├── compute_bimodal_time_experiment.py
    ├── compute_student_t_mixture.py
    ├── compute_student_t_score_network_kl.py
    ├── generate_elliptic_pde_data.py
    ├── compute_elliptic_pde_diffusion.py
    ├── compute_elliptic_pde_mcmc.py
    ├── environments/
    │   ├── pytorch_gpu_env.yml
    │   └── fenics_env.yml
    ├── plot_code/
    │   ├── plot_bimodal_c1_c9.py
    │   ├── plot_bimodal_time_experiment.py
    │   ├── plot_student_t_mixture.py
    │   └── plot_elliptic_pde.py
    └── slurm/
        ├── submit_bimodal_c1_c9.sh
        ├── run_plot_bimodal_c1_c9.sh
        ├── run_bimodal_time_experiment.sh
        ├── submit_all.sh
        ├── submit_student_t_mixture.sh
        ├── run_student_t_mixture_case.sh
        ├── submit_student_t_score_network_kl.sh
        ├── submit_elliptic_pde.sh
        ├── run_elliptic_pde_data.sh
        ├── run_elliptic_pde_diffusion.sh
        ├── run_elliptic_pde_mcmc.sh
        └── run_elliptic_pde_plots.sh

## Environments
- GPU: `conda env create --file environments/pytorch_gpu_env.yml`
- FEniCS: `conda env create --file environments/fenics_env.yml`

## Run
- Everything: `./slurm/submit_all.sh`
- Bimodal C1--C9: `./slurm/submit_bimodal_c1_c9.sh`
- Bimodal time experiment: `sbatch slurm/run_bimodal_time_experiment.sh`
- Elliptic PDE: `./slurm/submit_elliptic_pde.sh`
- Student-t mixture: `./slurm/submit_student_t_mixture.sh`
- Student-t score-network comparison: `./slurm/submit_student_t_score_network_kl.sh`
  
## Bimodal outputs
The C1--C9 experiment saves `C1.npz` through `C9.npz` and
`summary_table.tex` in `bimodal_c1_c9_results/`. Its figures are
`fig1_bimodal.pdf` and `fig1_bimodal.png` in `bimodal_c1_c9_results/figures/`.

The time experiment saves `time_experiment_results.npz` in `time_results/`.
Its figures are `fig2_time.pdf` and `fig2_time.png` in
`time_results/figures/`.

## Student-t mixture outputs
The six cases use variances 0.01, 0.1, and 0.5 for dimensions 10/10 and 15/5.
The figures are:
- `fig3_20D_i`
- `fig3_20D_ii`
- `fig4_2D_marginals_i`
- `fig4_2D_marginals_ii`
- `dim15_ablation`
- `dim10_ablation`

The score-network launcher submits one 10/10 job and one 15/5 job. The
projection KL and average marginal KL results are saved in:
- `student_t_score_network_results/dim10_10/projection_kl.csv`
- `student_t_score_network_results/dim10_10/average_marginal_kl.csv`
- `student_t_score_network_results/dim15_5/projection_kl.csv`
- `student_t_score_network_results/dim15_5/average_marginal_kl.csv`

## Elliptic PDE outputs
The elliptic PDE workflow runs data generation, diffusion, MCMC, and plotting
in the required order. Results are saved in `elliptic_pde_results/`.

The MCMC samples are:
- `MCMC_samples_testcase1.npy`
- `MCMC_samples_testcase2.npy`

The PDE figures are:
- `fig5_a` and `fig5_b`
- `fig6_pde_combined`
- `fig7_a` and `fig7_b`
- `fig8`
- `testcase1_ablations_varU` and `testcase2_ablations_varU`
- `testcase1_ablations_varV` and `testcase2_ablations_varV`


The supplied Slurm scripts use the `batch` and `batch-gpu` partitions and
request V100 GPUs. Adjust the partition, GPU, memory, and environment paths
for other computing systems.




