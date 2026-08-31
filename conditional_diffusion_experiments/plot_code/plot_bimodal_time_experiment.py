#!/usr/bin/env python3
"""Plot the saved bimodal time-discretization results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("time_results/time_experiment_results.npz"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)
    output_dir = args.output_dir or args.input_file.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_file, allow_pickle=False) as result:
        time_steps = result["time_steps"].astype(int)
        errors = result["epsilon_bgmm"].reshape(-1)

    if time_steps.shape != errors.shape:
        raise ValueError("time_steps and epsilon_bgmm have different shapes")
    if np.any(errors <= 0.0):
        raise ValueError("epsilon_bgmm must be positive for a logarithmic plot")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        time_steps,
        errors,
        "r.-",
        linewidth=2,
        markersize=10,
        label="Error",
    )
    axis.plot(
        time_steps,
        0.5 / time_steps,
        "k--",
        linewidth=2,
        label="Slope = -1",
    )
    axis.set_ylim(1.0e-3, 0.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(time_steps)
    axis.set_xticklabels([str(step) for step in time_steps])
    axis.minorticks_off()
    axis.set_xlabel("Timestep", fontsize=12)
    axis.set_ylabel("$e_{BGMM}$", fontsize=12)
    axis.grid(which="both", linestyle="--", linewidth=0.5)
    axis.legend(loc="upper right", fontsize=12)
    axis.set_title("Convergence of the $e_{BGMM}$", fontsize=16)
    figure.tight_layout()

    png_path = output_dir / "fig2_time.png"
    pdf_path = output_dir / "fig2_time.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
