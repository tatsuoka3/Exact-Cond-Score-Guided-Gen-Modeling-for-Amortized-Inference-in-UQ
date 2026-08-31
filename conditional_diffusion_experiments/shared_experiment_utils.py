#!/usr/bin/env python3
"""Reusable numerical helpers for conditional diffusion experiments."""

from __future__ import annotations

import numpy as np
import torch


EPS_ALPHA = 1.0e-5
EPS_BETA = 1.0e-5
KL_FLOOR = 1.0e-12

def select_device(name: str) -> torch.device:
    """Resolve a requested CPU, CUDA, or automatic PyTorch device."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable")
    return device


def standardize_variance(
    physical_variance: float | torch.Tensor,
    standard_deviation: torch.Tensor,
    *,
    full_like: bool = False,
) -> torch.Tensor:
    """Convert physical variance to coordinatewise standardized variance."""
    numerator = (
        torch.full_like(standard_deviation, physical_variance)
        if full_like
        else physical_variance
    )
    return numerator / standard_deviation.square()


def cond_alpha(t):
    return 1 - (1 - EPS_ALPHA) * t


def cond_beta2(t):
    return EPS_BETA + (1 - EPS_BETA) * t


def b(t):
    return -(1 - EPS_ALPHA) / cond_alpha(t)


def sigma_sq(t):
    return (1 - EPS_BETA) - 2 * b(t) * cond_beta2(t)


def sigma(t):
    return np.sqrt(sigma_sq(t))


def s1(t, variance):
    return cond_beta2(t) / (
        cond_alpha(t) ** 2 * variance + cond_beta2(t)
    )


def s2(t, variance):
    return cond_alpha(t) * variance / (
        cond_alpha(t) ** 2 * variance + cond_beta2(t)
    )


def s3(t, variance):
    return cond_beta2(t) * variance / (
        cond_alpha(t) ** 2 * variance + cond_beta2(t)
    )


def conditional_weights(
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """Normalize conditional log weights across empirical components."""
    return torch.softmax(log_weights, dim=1)


def conditional_score(
    z_t: torch.Tensor,
    t: float,
    sample_u: torch.Tensor,
    sample_v: torch.Tensor,
    condition_y: torch.Tensor,
    variance_u: torch.Tensor,
    variance_v: torch.Tensor,
    variance_y: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the empirical conditional diffusion score.

    condition_y may contain one condition shared by the whole batch with
    shape (dim_v,) or one condition per evaluation point with shape
    (n_eval, dim_v).
    """
    if condition_y.ndim == 1:
        condition_y = condition_y[None, :]
    if condition_y.ndim != 2 or condition_y.shape[0] not in (1, z_t.shape[0]):
        raise ValueError(
            "condition_y must have shape (dim_v,) or (n_eval, dim_v)"
        )

    dim_u = sample_u.shape[1]
    joint_samples = torch.cat((sample_u, sample_v), dim=1)
    joint_variances = torch.cat((variance_u, variance_v), dim=0)

    means_t = joint_samples * cond_alpha(t)
    variances_t = cond_beta2(t) + cond_alpha(t) ** 2 * joint_variances
    differences = z_t[:, None, :] - means_t[None, :, :]

    component_scores = -differences / variances_t
    log_weights = 0.5 * torch.sum(
        component_scores * differences,
        dim=2,
    )

    y_difference = (
        z_t[:, None, dim_u:] * s2(t, variance_v)
        + sample_v[None, :, :] * s1(t, variance_v)
        - condition_y[:, None, :]
    )
    likelihood_score = -y_difference / (
        s3(t, variance_v) + variance_y
    )
    component_scores[:, :, dim_u:] = (
        component_scores[:, :, dim_u:]
        + likelihood_score * s2(t, variance_v)
    )
    log_weights += 0.5 * torch.sum(
        likelihood_score * y_difference,
        dim=2,
    )

    weights = conditional_weights(log_weights)
    return torch.sum(component_scores * weights[:, :, None], dim=1)


def reverse_sde(
    x_terminal: torch.Tensor,
    time_steps: int,
    drift,
    diffusion,
    score,
    *,
    save_path: bool = True,
):
    """Solve the reverse SDE with the original forward-Euler update."""
    dt = 1.0 / time_steps
    x_t = x_terminal.clone()
    t = 1.0
    times = [t]
    path = [x_t]

    for _ in range(time_steps):
        diffuse = diffusion(t)
        reverse_drift = drift(t) * x_t - diffuse**2 * score(x_t, t) / 2
        x_t = x_t - dt * reverse_drift
        if save_path:
            path.append(x_t)
        times.append(t)
        t = t - dt

    if save_path:
        return path, times
    return x_t


def trapezoid_integral(values: np.ndarray, grid: np.ndarray) -> float:
    """Integrate sampled values on a grid."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))
    return float(np.trapz(values, grid))


def normalize_density(
    density: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Normalize a density on a finite grid."""
    integral = trapezoid_integral(density, grid)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError(f"Invalid density integral: {integral}")
    return density / integral


def kl_divergence(
    target: np.ndarray,
    approximation: np.ndarray,
    grid: np.ndarray,
    *,
    floor: float = KL_FLOOR,
) -> float:
    """Compute D_KL(target || approximation) on an evenly spaced grid."""
    target_safe = np.clip(target, floor, None)
    approximation_safe = np.clip(approximation, floor, None)
    dx = float(grid[1] - grid[0])
    return float(
        np.sum(target_safe * np.log(target_safe / approximation_safe)) * dx
    )
