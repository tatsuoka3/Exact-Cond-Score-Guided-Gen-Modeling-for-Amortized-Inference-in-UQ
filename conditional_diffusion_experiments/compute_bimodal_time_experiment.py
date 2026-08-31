#!/usr/bin/env python3
"""Compute and save the bimodal time-discretization experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bimodal_experiment import (
    CONFIGURATIONS,
    N_GENERATED,
    SCORE_BATCH_SIZE,
    build_dataset,
    compare_generated_samples,
    generate_conditional_samples,
    make_grid,
    reference_densities,
)
from shared_experiment_utils import select_device


DEFAULT_TIME_STEPS = (8, 16, 32, 64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("time_results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--config-index", type=int, choices=range(1, 10), default=5)
    parser.add_argument("--n-generated", type=int, default=N_GENERATED)
    parser.add_argument("--score-batch-size", type=int, default=SCORE_BATCH_SIZE)
    parser.add_argument(
        "--time-steps",
        type=int,
        nargs="+",
        default=DEFAULT_TIME_STEPS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_generated < 2 or args.score_batch_size < 1:
        raise ValueError("sample count must be at least 2 and batch size positive")
    if any(steps < 1 for steps in args.time_steps):
        raise ValueError("all time-step counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "time_experiment_results.npz"
    config = CONFIGURATIONS[args.config_index - 1]
    device = select_device(args.device)
    started = time.perf_counter()

    dataset = build_dataset(config, device, args.n_generated)
    grid = make_grid()
    references = reference_densities(grid, dataset, config)
    bandwidth_grid = np.linspace(0.12, 0.04, len(args.time_steps))

    generated_by_time = []
    comparisons = []
    for steps, bandwidth in zip(args.time_steps, bandwidth_grid):
        print(f"Running {steps} Euler steps", flush=True)
        generated = generate_conditional_samples(
            dataset,
            config,
            steps,
            args.score_batch_size,
        )
        generated_by_time.append(generated)
        comparisons.append(
            compare_generated_samples(
                generated[:, 0],
                grid,
                references,
                float(bandwidth),
            )
        )

    np.savez(
        output_path,
        config_label=np.array(config.label),
        config_index=np.array(args.config_index),
        K=np.array(config.K),
        sigma_u_squared=np.array(config.sigma_u_squared),
        sigma_v_squared=np.array(config.sigma_v_squared),
        sigma_y_squared=np.array(config.sigma_y_squared),
        time_steps=np.asarray(args.time_steps, dtype=int),
        generated_samples=np.stack(generated_by_time),
        sample_u=dataset.sample_u.cpu().numpy(),
        sample_v=dataset.sample_v.cpu().numpy(),
        grid=grid,
        exact_density=references.exact,
        gmm_density=references.gmm,
        bgmm_density=references.bgmm,
        epsilon_exact=np.array([row.epsilon_exact for row in comparisons]),
        epsilon_gmm=np.array([row.epsilon_gmm for row in comparisons]),
        epsilon_bgmm=np.array([row.epsilon_bgmm for row in comparisons]),
        kde_densities=np.stack([row.density for row in comparisons]),
        n_generated=np.array(args.n_generated),
        elapsed_seconds=np.array(time.perf_counter() - started),
    )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
