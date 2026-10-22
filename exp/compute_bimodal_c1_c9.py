#!/usr/bin/env python3
"""Compute and save the C1--C9 bimodal experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bimodal_experiment import (
    CONFIGURATIONS,
    N_GENERATED,
    SCORE_BATCH_SIZE,
    TIME_STEPS,
    build_dataset,
    compare_generated_samples,
    generate_conditional_samples,
    make_grid,
    reference_densities,
    standardized_variances,
)
from shared_experiment_utils import select_device


BANDWIDTHS = np.arange(0.01, 0.101, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-index",
        type=int,
        choices=range(1, len(CONFIGURATIONS) + 1),
        help="Run one configuration; omit to run all nine.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bimodal_c1_c9_results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--n-generated", type=int, default=N_GENERATED)
    parser.add_argument("--time-steps", type=int, default=TIME_STEPS)
    parser.add_argument("--score-batch-size", type=int, default=SCORE_BATCH_SIZE)
    return parser.parse_args()


def run_configuration(args: argparse.Namespace, config_index: int) -> None:
    config = CONFIGURATIONS[config_index - 1]
    device = select_device(args.device)
    started = time.perf_counter()

    print(f"Running {config.label} on {device}", flush=True)
    dataset = build_dataset(config, device, args.n_generated)
    score_variances = standardized_variances(dataset, config)
    generated = generate_conditional_samples(
        dataset,
        config,
        args.time_steps,
        args.score_batch_size,
    )

    grid = make_grid()
    references = reference_densities(grid, dataset, config)
    comparisons = [
        compare_generated_samples(generated[:, 0], grid, references, float(h))
        for h in BANDWIDTHS
    ]
    epsilon_bgmm = np.array([row.epsilon_bgmm for row in comparisons])
    best_index = int(np.argmin(epsilon_bgmm))

    output_path = args.output_dir / f"{config.label}.npz"
    np.savez(
        output_path,
        config_label=np.array(config.label),
        config_index=np.array(config_index),
        K=np.array(config.K),
        sigma_u_squared=np.array(config.sigma_u_squared),
        sigma_v_squared=np.array(config.sigma_v_squared),
        sigma_y_squared=np.array(config.sigma_y_squared),
        empirical_mean_u=dataset.mean_u.cpu().numpy(),
        empirical_std_u=dataset.std_u.cpu().numpy(),
        empirical_mean_v=dataset.mean_v.cpu().numpy(),
        empirical_std_v=dataset.std_v.cpu().numpy(),
        standardized_sigma_u_squared=score_variances.sigma_u_squared.cpu().numpy(),
        standardized_sigma_v_squared=score_variances.sigma_v_squared.cpu().numpy(),
        standardized_sigma_y_squared=score_variances.sigma_y_squared.cpu().numpy(),
        sample_u=dataset.sample_u.cpu().numpy(),
        sample_v=dataset.sample_v.cpu().numpy(),
        generated_samples=generated,
        grid=grid,
        exact_density=references.exact,
        gmm_density=references.gmm,
        bgmm_density=references.bgmm,
        bandwidths=BANDWIDTHS,
        kde_sigmas=np.array([row.kernel_sigma for row in comparisons]),
        epsilon_exact=np.array([row.epsilon_exact for row in comparisons]),
        epsilon_gmm=np.array([row.epsilon_gmm for row in comparisons]),
        epsilon_bgmm=epsilon_bgmm,
        h_best=np.array(BANDWIDTHS[best_index]),
        best_kde_density=comparisons[best_index].density,
        n_generated=np.array(args.n_generated),
        time_steps=np.array(args.time_steps),
        elapsed_seconds=np.array(time.perf_counter() - started),
    )

    print(
        f"Saved {output_path}; h_best={BANDWIDTHS[best_index]:.2f}, "
        f"e_BGMM={epsilon_bgmm[best_index]:.6e}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.n_generated < 2:
        raise ValueError("--n-generated must be at least 2")
    if args.time_steps < 1 or args.score_batch_size < 1:
        raise ValueError("time steps and batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indices = (
        [args.config_index]
        if args.config_index is not None
        else range(1, len(CONFIGURATIONS) + 1)
    )
    for config_index in indices:
        run_configuration(args, config_index)


if __name__ == "__main__":
    main()
