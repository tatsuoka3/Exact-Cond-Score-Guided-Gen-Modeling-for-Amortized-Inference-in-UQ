#!/usr/bin/env python3
"""Definition and reference densities for the one-dimensional bimodal study."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import gaussian_kde

from shared_experiment_utils import (
    b,
    conditional_score,
    kl_divergence,
    normalize_density,
    sigma_sq,
    standardize_variance,
)


SEED = 1234
DIM_U = 1
DIM_V = 1
Y_OBS = 1.0
DATA_NOISE_VARIANCE = 0.1
N_GENERATED = 10_000
TIME_STEPS = 1_000
SCORE_BATCH_SIZE = 1_000
GRID_LEFT = -4.0
GRID_RIGHT = 4.0
GRID_SIZE = 1_000


@dataclass(frozen=True)
class Configuration:
    label: str
    K: int
    sigma_u_squared: float
    sigma_y_squared: float

    @property
    def sigma_v_squared(self) -> float:
        return self.sigma_u_squared


CONFIGURATIONS = (
    Configuration("C1", 500, 0.005, 0.0001),
    Configuration("C2", 500, 0.010, 0.0001),
    Configuration("C3", 500, 0.050, 0.0001),
    Configuration("C4", 5000, 0.001, 0.0001),
    Configuration("C5", 5000, 0.005, 0.0001),
    Configuration("C6", 5000, 0.010, 0.0001),
    Configuration("C7", 5000, 0.005, 0.0010),
    Configuration("C8", 5000, 0.005, 0.0100),
    Configuration("C9", 5000, 0.005, 0.1000),
)


@dataclass
class Dataset:
    sample_u: torch.Tensor
    sample_v: torch.Tensor
    sample_u_standardized: torch.Tensor
    sample_v_standardized: torch.Tensor
    mean_u: torch.Tensor
    std_u: torch.Tensor
    mean_v: torch.Tensor
    std_v: torch.Tensor
    observed_y_standardized: torch.Tensor
    terminal_samples: torch.Tensor


@dataclass(frozen=True)
class StandardizedVariances:
    sigma_u_squared: torch.Tensor
    sigma_v_squared: torch.Tensor
    sigma_y_squared: torch.Tensor


@dataclass(frozen=True)
class ReferenceDensities:
    exact: np.ndarray
    gmm: np.ndarray
    bgmm: np.ndarray


@dataclass(frozen=True)
class KDEComparison:
    density: np.ndarray
    bandwidth_factor: float
    kernel_sigma: float
    epsilon_exact: float
    epsilon_gmm: float
    epsilon_bgmm: float


def make_grid() -> np.ndarray:
    return np.linspace(GRID_LEFT, GRID_RIGHT, GRID_SIZE)


def build_dataset(
    config: Configuration,
    device: torch.device,
    n_generated: int,
    seed: int = SEED,
) -> Dataset:
    """Create training data and terminal samples in the original RNG order."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    sample_u = -2.0 + 4.0 * torch.rand(
        config.K,
        DIM_U,
        device=device,
        dtype=torch.float32,
    )
    sample_v = sample_u.square() + torch.randn(
        config.K,
        DIM_V,
        device=device,
        dtype=torch.float32,
    ) * math.sqrt(DATA_NOISE_VARIANCE)

    mean_u = sample_u.mean(dim=0)
    std_u = sample_u.std(dim=0)
    mean_v = sample_v.mean(dim=0)
    std_v = sample_v.std(dim=0)
    if torch.any(std_u <= 0.0) or torch.any(std_v <= 0.0):
        raise RuntimeError("Cannot standardize data with zero variance")

    observed_y = torch.tensor([Y_OBS], device=device, dtype=torch.float32)
    return Dataset(
        sample_u=sample_u,
        sample_v=sample_v,
        sample_u_standardized=(sample_u - mean_u) / std_u,
        sample_v_standardized=(sample_v - mean_v) / std_v,
        mean_u=mean_u,
        std_u=std_u,
        mean_v=mean_v,
        std_v=std_v,
        observed_y_standardized=(observed_y - mean_v) / std_v,
        terminal_samples=torch.randn(
            n_generated,
            DIM_U + DIM_V,
            device=device,
            dtype=torch.float32,
        ),
    )


def standardized_variances(
    dataset: Dataset,
    config: Configuration,
) -> StandardizedVariances:
    """Package physical variances in standardized score coordinates."""
    return StandardizedVariances(
        sigma_u_squared=standardize_variance(
            config.sigma_u_squared,
            dataset.std_u,
            full_like=True,
        ),
        sigma_v_squared=standardize_variance(
            config.sigma_v_squared,
            dataset.std_v,
            full_like=True,
        ),
        sigma_y_squared=standardize_variance(
            config.sigma_y_squared,
            dataset.std_v,
            full_like=True,
        ),
    )


@torch.inference_mode()
def generate_conditional_samples(
    dataset: Dataset,
    config: Configuration,
    time_steps: int,
    score_batch_size: int,
) -> np.ndarray:
    """Generate conditional samples with the original Euler update."""
    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    if score_batch_size < 1:
        raise ValueError("score_batch_size must be positive")

    variances = standardized_variances(dataset, config)
    dt = 1.0 / time_steps
    batches: list[np.ndarray] = []

    for start in range(0, dataset.terminal_samples.shape[0], score_batch_size):
        stop = min(
            start + score_batch_size,
            dataset.terminal_samples.shape[0],
        )
        x_t = dataset.terminal_samples[start:stop].clone()
        t = 1.0

        for _ in range(time_steps):
            score = conditional_score(
                x_t,
                t,
                sample_u=dataset.sample_u_standardized,
                sample_v=dataset.sample_v_standardized,
                condition_y=dataset.observed_y_standardized,
                variance_u=variances.sigma_u_squared,
                variance_v=variances.sigma_v_squared,
                variance_y=variances.sigma_y_squared,
            )
            reverse_drift = b(t) * x_t - 0.5 * sigma_sq(t) * score
            x_t -= dt * reverse_drift
            t -= dt

        x_t[:, 0] = x_t[:, 0] * dataset.std_u + dataset.mean_u
        x_t[:, 1] = x_t[:, 1] * dataset.std_v + dataset.mean_v
        batches.append(x_t.cpu().numpy())
        print(
            f"    generated {stop}/{dataset.terminal_samples.shape[0]} samples",
            flush=True,
        )

    return np.concatenate(batches, axis=0)


def exact_posterior_density(grid: np.ndarray) -> np.ndarray:
    def unnormalized(u: float) -> float:
        return math.exp(
            -((Y_OBS - u * u) ** 2) / (2.0 * DATA_NOISE_VARIANCE)
        )

    denominator = quad(unnormalized, -np.inf, np.inf)[0]
    return np.exp(
        -((Y_OBS - grid**2) ** 2) / (2.0 * DATA_NOISE_VARIANCE)
    ) / denominator


def conditional_gmm_density(
    grid: np.ndarray,
    dataset: Dataset,
    config: Configuration,
    bayesian: bool,
    component_batch_size: int = 1_000,
) -> np.ndarray:
    sample_u = dataset.sample_u[:, 0].cpu().numpy().astype(np.float64)
    sample_v = dataset.sample_v[:, 0].cpu().numpy().astype(np.float64)
    conditioning_variance = config.sigma_v_squared
    if bayesian:
        conditioning_variance += config.sigma_y_squared

    log_likelihoods = -0.5 * (
        (Y_OBS - sample_v) ** 2 / conditioning_variance
        + math.log(2.0 * math.pi * conditioning_variance)
    )
    log_likelihoods -= np.max(log_likelihoods)
    weights = np.exp(log_likelihoods)
    weights /= weights.sum()

    normalizer = math.sqrt(2.0 * math.pi * config.sigma_u_squared)
    density = np.zeros_like(grid, dtype=np.float64)
    for start in range(0, config.K, component_batch_size):
        stop = min(start + component_batch_size, config.K)
        differences = grid[:, None] - sample_u[None, start:stop]
        components = np.exp(
            -0.5 * differences**2 / config.sigma_u_squared
        ) / normalizer
        density += components @ weights[start:stop]
    return density


def reference_densities(
    grid: np.ndarray,
    dataset: Dataset,
    config: Configuration,
) -> ReferenceDensities:
    return ReferenceDensities(
        exact=normalize_density(exact_posterior_density(grid), grid),
        gmm=normalize_density(
            conditional_gmm_density(grid, dataset, config, bayesian=False),
            grid,
        ),
        bgmm=normalize_density(
            conditional_gmm_density(grid, dataset, config, bayesian=True),
            grid,
        ),
    )


def compare_generated_samples(
    generated_u: np.ndarray,
    grid: np.ndarray,
    references: ReferenceDensities,
    bandwidth: float | None,
) -> KDEComparison:
    kde = gaussian_kde(generated_u, bw_method=bandwidth)
    kde_density = normalize_density(kde(grid), grid)
    return KDEComparison(
        density=kde_density,
        bandwidth_factor=float(kde.factor),
        kernel_sigma=float(np.sqrt(kde.covariance[0, 0])),
        epsilon_exact=kl_divergence(references.exact, kde_density, grid),
        epsilon_gmm=kl_divergence(references.gmm, kde_density, grid),
        epsilon_bgmm=kl_divergence(references.bgmm, kde_density, grid),
    )
