#!/usr/bin/env python3
"""Plot the Student-t mixture results for U/V = 10/10 and 15/5.

The script reads the six result directories created by the Slurm sweep. It
creates, for each dimensional split:

1. a 2 x 3 NN/DM projection comparison at physical variance 0.1;
2. a 2 x 3 true-versus-NN 2D KDE marginal comparison at y = 0;
3. a 3 x 3 DM physical-variance sweep for 0.01, 0.1, and 0.5.

Both PNG and PDF versions are saved.  Projection KL values use the same
finite-grid normalization, KDE, and D_KL(p_true || p_model) direction as the
training scripts and reference notebook.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import gammaln
from scipy.stats import gaussian_kde, t as student_t

from shared_experiment_utils import kl_divergence, normalize_density


TOTAL_DIM = 20
NU_TRUE = 10.0
SIGMA_VALUE = 1.0
MIXTURE_WEIGHTS = np.array([0.5, 0.5], dtype=float)

# Display order used by the supplied notebook (the arrays themselves are in
# the order -0.25, 0.0, +0.25).
CONDITION_ORDER = (0.0, 0.25, -0.25)
VARIANCE_ORDER = (0.01, 0.1, 0.5)

PROJECTION_GRID = np.linspace(-10.0, 10.0, 500)
PROJECTION_BINS = np.linspace(-10.0, 10.0, 21)
SWEEP_BINS = np.linspace(-10.0, 10.0, 52)
PLOT_RANGE = (-10.0, 10.0)


def component_means() -> tuple[np.ndarray, np.ndarray]:
    """Return the two 20D component locations used by the experiments."""
    mu1 = np.zeros(TOTAL_DIM, dtype=float)
    mu2 = np.zeros(TOTAL_DIM, dtype=float)
    for indices, delta in (
        (range(0, 5), 1.35),
        (range(5, 10), 0.50),
        (range(10, 15), 0.20),
        (range(15, 20), 0.10),
    ):
        idx = list(indices)
        mu1[idx] = -delta
        mu2[idx] = +delta
    return mu1, mu2


MU1, MU2 = component_means()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Student-t mixture 10/10 and 15/5 sweep."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "student_t_mixture_results",
        help="Root containing the six variance/dimension result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory receiving PDF, PNG, and CSV outputs.",
    )
    parser.add_argument(
        "--marginal-condition",
        type=float,
        default=0.0,
        choices=(-0.25, 0.0, 0.25),
        help="Condition used in the true-versus-model 2D KDE plots.",
    )
    parser.add_argument(
        "--marginal-model",
        choices=("NN", "DM", "both"),
        default="NN",
        help="Model shown below the truth in the 2D KDE plots.",
    )
    parser.add_argument(
        "--dim-u",
        type=int,
        choices=(10, 15),
        default=None,
        help="Plot only one dimensional split; defaults to both.",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Generate only the DM variance-ablation plot.",
    )
    return parser.parse_args()


def load_result(
    input_dir: Path,
    dim_u: int,
    variance: float,
) -> dict[str, np.ndarray]:
    dim_v = TOTAL_DIM - dim_u
    variance_label = f"{float(variance):g}".replace(".", "p")
    input_path = (
        input_dir
        / f"var{variance_label}_{dim_u}u{dim_v}v"
        / "student_t_mixture_samples_all_conditions.npz"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Student-t mixture result not found: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def condition_index(data: dict[str, np.ndarray], condition: float) -> int:
    hits = np.flatnonzero(np.isclose(data["conditions"], condition))
    if hits.size != 1:
        raise ValueError(
            f"Could not uniquely find condition {condition} in {data['conditions']!r}."
        )
    return int(hits[0])


def multivariate_t_logpdf_isotropic(x: np.ndarray, mu: np.ndarray) -> float:
    """Log density for t_nu(mu, SIGMA_VALUE^2 I)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    dimension = x.size
    delta = np.sum((x - mu) ** 2) / SIGMA_VALUE**2
    return float(
        gammaln((NU_TRUE + dimension) / 2.0)
        - gammaln(NU_TRUE / 2.0)
        - 0.5 * dimension * np.log(NU_TRUE * np.pi)
        - dimension * np.log(SIGMA_VALUE)
        - 0.5 * (NU_TRUE + dimension) * np.log1p(delta / NU_TRUE)
    )


def conditional_mixture_parameters(
    condition: float,
    dim_u: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return weights, df, U locations, and scalar scales for U | V=y1."""
    dim_v = TOTAL_DIM - dim_u
    y = float(condition) * np.ones(dim_v, dtype=float)
    locations = np.stack((MU1[:dim_u], MU2[:dim_u]))
    v_locations = (MU1[dim_u:], MU2[dim_u:])

    log_weights = np.array(
        [
            np.log(MIXTURE_WEIGHTS[k])
            + multivariate_t_logpdf_isotropic(y, v_locations[k])
            for k in range(2)
        ]
    )
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()

    deltas = np.array(
        [np.sum((y - v_locations[k]) ** 2) / SIGMA_VALUE**2 for k in range(2)]
    )
    scales = np.sqrt(
        SIGMA_VALUE**2 * (NU_TRUE + deltas) / (NU_TRUE + dim_v)
    )
    return weights, NU_TRUE + dim_v, locations, scales


def true_1d_mixture_density(
    grid: np.ndarray,
    weights: np.ndarray,
    degrees_freedom: float,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    density = np.zeros_like(grid, dtype=float)
    for component in range(2):
        density += (
            weights[component]
            * student_t.pdf(
                (grid - locations[component]) / scales[component],
                df=degrees_freedom,
            )
            / scales[component]
        )
    return density


def projection_result(
    samples: np.ndarray,
    condition: float,
    dim_u: int,
) -> dict[str, np.ndarray | float]:
    """Calculate exact projected truth, sample projection, KDE, and KL."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != dim_u:
        raise ValueError(f"Expected samples with shape (N, {dim_u}); got {samples.shape}.")

    weights, conditional_nu, locations, scales = conditional_mixture_parameters(
        condition, dim_u
    )
    direction = locations[1] - locations[0]
    direction /= np.linalg.norm(direction) + 1e-14

    projected_locations = locations @ direction
    truth = true_1d_mixture_density(
        PROJECTION_GRID,
        weights,
        conditional_nu,
        projected_locations,
        scales,
    )
    truth = normalize_density(truth, PROJECTION_GRID)

    projected = samples @ direction
    kde = normalize_density(
        gaussian_kde(projected)(PROJECTION_GRID),
        PROJECTION_GRID,
    )
    divergence = kl_divergence(
        truth,
        kde,
        PROJECTION_GRID,
    )
    return {
        "truth": truth,
        "projected": projected,
        "kde": kde,
        "kl": divergence,
    }


def sample_conditional_truth(
    condition: float,
    dim_u: int,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Sample the exact conditional Student-t mixture, preserving dependence."""
    weights, conditional_nu, locations, scales = conditional_mixture_parameters(
        condition, dim_u
    )
    rng = np.random.default_rng(seed)
    labels = rng.choice(2, size=sample_count, p=weights)
    result = np.empty((sample_count, dim_u), dtype=float)
    for component in range(2):
        rows = np.flatnonzero(labels == component)
        if rows.size == 0:
            continue
        normal = rng.standard_normal((rows.size, dim_u))
        chi_square = rng.chisquare(conditional_nu, size=rows.size)
        radial = np.sqrt(conditional_nu / chi_square)[:, None]
        result[rows] = locations[component] + scales[component] * normal * radial
    return result


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def histogram_peak(values: np.ndarray, bins: np.ndarray) -> float:
    heights, _ = np.histogram(values, bins=bins, density=True)
    return float(np.max(heights))


def plot_nn_dm_projection_comparison(
    data: dict[str, np.ndarray],
    dim_u: int,
    output_dir: Path,
) -> list[dict[str, float | str | int]]:
    """Notebook Part A: NN row, DM row, with y=(0,+0.25,-0.25)."""
    dim_v = TOTAL_DIM - dim_u
    figure_name = {15: "fig3_20D_i", 10: "fig3_20D_ii"}[dim_u]
    model_specs = (("NN", "U_nn_phys", "blue"), ("DM", "U_dm_phys", "red"))
    cache: dict[tuple[str, float], dict[str, np.ndarray | float]] = {}
    rows: list[dict[str, float | str | int]] = []

    for model_name, array_key, _ in model_specs:
        for condition in CONDITION_ORDER:
            idx = condition_index(data, condition)
            result = projection_result(data[array_key][idx], condition, dim_u)
            cache[(model_name, condition)] = result
            rows.append(
                {
                    "figure": figure_name,
                    "dim_u": dim_u,
                    "dim_v": dim_v,
                    "variance": 0.1,
                    "model": model_name,
                    "condition": condition,
                    "projection_KL": float(result["kl"]),
                }
            )

    column_maxima: list[float] = []
    for condition in CONDITION_ORDER:
        maximum = 0.0
        for model_name, _, _ in model_specs:
            result = cache[(model_name, condition)]
            maximum = max(
                maximum,
                float(np.max(result["truth"])),
                histogram_peak(np.asarray(result["projected"]), PROJECTION_BINS),
            )
        column_maxima.append(maximum)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), squeeze=False)
    for row, (model_name, _, color) in enumerate(model_specs):
        for col, condition in enumerate(CONDITION_ORDER):
            ax = axes[row, col]
            result = cache[(model_name, condition)]
            line = ax.plot(
                PROJECTION_GRID,
                result["truth"],
                color="green",
                linewidth=1.5,
                alpha=0.9,
            )[0]
            _, _, patches = ax.hist(
                result["projected"],
                bins=PROJECTION_BINS,
                density=True,
                alpha=0.5,
                color=color,
            )
            if row == 0 and col == 0:
                line.set_label("True Density")
                patches[0].set_label("Samples (NN)")
            if row == 1 and col == 0:
                patches[0].set_label("Samples (DM)")

            condition_suffix = rf"$\cdot \mathbf{{1}}_{{{dim_v}}}$"
            ax.set_title(
                f"{model_name}: $y$ = {condition:g} {condition_suffix}",
                fontsize=14,
            )
            ax.set_xlabel(r"$U$", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.grid(which="both", linestyle="--", linewidth=0.5)
            ax.set_xlim(*PLOT_RANGE)
            ax.set_ylim(0.0, 1.05 * column_maxima[col])

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), h_pad=0.2, w_pad=1.0)
    save_figure(
        fig,
        output_dir / figure_name,
    )
    return rows


def plot_2d_marginals(
    data: dict[str, np.ndarray],
    dim_u: int,
    condition: float,
    model_name: str,
    output_dir: Path,
) -> None:
    """Notebook Part B: true KDE row and model KDE row."""
    dim_v = TOTAL_DIM - dim_u
    array_key = {"NN": "U_nn_phys", "DM": "U_dm_phys"}[model_name]
    idx = condition_index(data, condition)
    model_samples = np.asarray(data[array_key][idx], dtype=float)
    true_samples = sample_conditional_truth(
        condition,
        dim_u,
        sample_count=model_samples.shape[0],
        seed=1234 + dim_u,
    )

    # These are the same zero-based coordinate pairs as the reference notebook.
    selected_pairs = ((1, 2), (4, 5), (5, 6))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), squeeze=False)
    for col, (i, j) in enumerate(selected_pairs):
        ax_true = axes[0, col]
        ax_model = axes[1, col]

        sns.kdeplot(
            x=true_samples[:, i],
            y=true_samples[:, j],
            cmap="Greens",
            fill=True,
            levels=10,
            ax=ax_true,
        )
        ax_true.set_title(rf"True: $u_{{{i}}}$ vs $u_{{{j}}}$", fontsize=18)
        ax_true.set_xlabel(rf"$u_{{{i}}}$", fontsize=16)
        ax_true.set_ylabel(rf"$u_{{{j}}}$", fontsize=16)

        sns.kdeplot(
            x=model_samples[:, i],
            y=model_samples[:, j],
            cmap="Blues" if model_name == "NN" else "Reds",
            fill=True,
            levels=10,
            ax=ax_model,
        )
        ax_model.set_title(
            rf"{model_name}: $u_{{{i}}}$ vs $u_{{{j}}}$", fontsize=18
        )
        ax_model.set_xlabel(rf"$u_{{{i}}}$", fontsize=16)
        ax_model.set_ylabel(rf"$u_{{{j}}}$", fontsize=16)

        for ax in (ax_true, ax_model):
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(-4.0, 4.0)
            ax.grid(which="both", linestyle="--", linewidth=0.5)

    fig.suptitle(
        rf"$U^{{{dim_u}}}\mid V={condition:g}\mathbf{{1}}_{{{dim_v}}}$",
        fontsize=15,
        y=0.97,
    )
    # Explicit spacing is more reliable than tight_layout for these large
    # math-text titles, especially for the U^15 | V^5 case.
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.09,
        top=0.84,
        wspace=0.36,
        hspace=0.58,
    )
    figure_name = {15: "fig4_2D_marginals_i", 10: "fig4_2D_marginals_ii"}[dim_u]
    if model_name != "NN" or condition != 0.0:
        condition_tag = (
            f"{condition:+.2f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )
        figure_name = f"{figure_name}_{model_name}_cond{condition_tag}"
    save_figure(
        fig,
        output_dir / figure_name,
    )


def plot_dm_variance_sweep(
    datasets: dict[float, dict[str, np.ndarray]],
    dim_u: int,
    output_dir: Path,
) -> list[dict[str, float | str | int]]:
    """Notebook sweep figure: variance rows and condition columns."""
    dim_v = TOTAL_DIM - dim_u
    figure_name = {15: "dim15_ablation", 10: "dim10_ablation"}[dim_u]
    cache: dict[tuple[float, float], dict[str, np.ndarray | float]] = {}
    rows: list[dict[str, float | str | int]] = []
    column_maxima = [0.0, 0.0, 0.0]

    for variance in VARIANCE_ORDER:
        data = datasets[variance]
        for col, condition in enumerate(CONDITION_ORDER):
            idx = condition_index(data, condition)
            result = projection_result(data["U_dm_phys"][idx], condition, dim_u)
            cache[(variance, condition)] = result
            column_maxima[col] = max(
                column_maxima[col],
                float(np.max(result["truth"])),
                histogram_peak(np.asarray(result["projected"]), SWEEP_BINS),
            )
            rows.append(
                {
                    "figure": figure_name,
                    "dim_u": dim_u,
                    "dim_v": dim_v,
                    "variance": variance,
                    "model": "DM",
                    "condition": condition,
                    "projection_KL": float(result["kl"]),
                }
            )

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.7), squeeze=False)
    for row, variance in enumerate(VARIANCE_ORDER):
        for col, condition in enumerate(CONDITION_ORDER):
            ax = axes[row, col]
            result = cache[(variance, condition)]
            line = ax.plot(
                PROJECTION_GRID,
                result["truth"],
                color="green",
                linewidth=1.5,
                alpha=0.9,
            )[0]
            _, _, patches = ax.hist(
                result["projected"],
                bins=SWEEP_BINS,
                density=True,
                alpha=0.5,
                color="red",
            )
            if row == 0 and col == 0:
                line.set_label("True Density")
                patches[0].set_label("Samples (DM)")

            condition_suffix = rf"$\cdot \mathbf{{1}}_{{{dim_v}}}$"
            ax.set_title(
                rf"$y$ = {condition:g} {condition_suffix}"
                + "\n"
                + rf"$\sigma_U^2={variance:g}$, KL = {float(result['kl']):.4f}",
                fontsize=11,
            )
            if col == 0:
                ax.set_ylabel("Density", fontsize=12)
            if row == len(VARIANCE_ORDER) - 1:
                ax.set_xlabel(r"$U$", fontsize=12)
            ax.grid(which="both", linestyle="--", linewidth=0.5)
            ax.set_xlim(*PLOT_RANGE)
            ax.set_ylim(0.0, 0.5)

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=11)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.96,
        wspace=0.33,
        hspace=0.78,
    )
    save_figure(
        fig,
        output_dir / figure_name,
    )
    return rows


def write_kl_summary(
    rows: list[dict[str, float | str | int]],
    output_path: Path,
) -> None:
    fieldnames = (
        "figure",
        "dim_u",
        "dim_v",
        "variance",
        "model",
        "condition",
        "projection_KL",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the notebook's uncluttered Matplotlib appearance.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": True,
        }
    )

    dimensions = (args.dim_u,) if args.dim_u is not None else (10, 15)
    all_rows: list[dict[str, float | str | int]] = []
    for dim_u in dimensions:
        dim_v = TOTAL_DIM - dim_u
        print(f"[load] dim_u={dim_u}, dim_v={dim_v}", flush=True)
        datasets = {
            variance: load_result(args.input_dir, dim_u, variance)
            for variance in VARIANCE_ORDER
        }
        if not args.ablation_only:
            full_data = datasets[0.1]
            all_rows.extend(
                plot_nn_dm_projection_comparison(full_data, dim_u, output_dir)
            )

            marginal_models = (
                ("NN", "DM")
                if args.marginal_model == "both"
                else (args.marginal_model,)
            )
            for model_name in marginal_models:
                plot_2d_marginals(
                    full_data,
                    dim_u,
                    args.marginal_condition,
                    model_name,
                    output_dir,
                )

        all_rows.extend(plot_dm_variance_sweep(datasets, dim_u, output_dir))

    if args.ablation_only:
        print(f"[saved] ablation figure in {output_dir}", flush=True)
    else:
        summary_path = output_dir / "projection_KL_summary.csv"
        write_kl_summary(all_rows, summary_path)
        print(f"[saved] figures and KL summary in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
