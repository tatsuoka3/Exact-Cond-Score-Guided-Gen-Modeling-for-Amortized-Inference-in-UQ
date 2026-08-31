#!/usr/bin/env python3
"""Train or load the Student-t score network and report KL metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gammaln
from scipy.stats import gaussian_kde, t as student_t

from shared_experiment_utils import kl_divergence, normalize_density, select_device


seed = 1234
condition_seed_stride = 10_007
total_dim = 20
conditions = (-0.25, 0.0, 0.25)
table_conditions = (0.0, -0.25, 0.25)

nu_t = 10.0
student_t_scale = 1.0
mixture_weights = np.array([0.5, 0.5], dtype=np.float64)

n_joint_samples = 250_000
n_training_samples = 50_000
n_test_samples = 5_000

width = 128
depth = 6
sigma_embedding_frequencies = 128
sigma_embedding_dim = 128
sigma_embedding_hidden = 256

batch_size = 4_096
epochs = 50_000
learning_rate = 1e-3
gradient_clip = 1.0
log_every = 1_000
ema_decay = 0.9999

sigma_max = 25.0
sigma_min = 0.01
noise_levels = 128
sigma_sampling_power = 2.0

langevin_steps_per_level = 5
langevin_step_size = 5.7e-6
score_clip = 20.0
sample_clip = 50.0
max_backoffs = 6

projection_grid = np.linspace(-10.0, 10.0, 500)
marginal_grid = np.linspace(-8.0, 8.0, 1_200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim-u", type=int, choices=(10, 15), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def condition_seed(condition: float) -> int:
    index = conditions.index(float(condition))
    return seed + condition_seed_stride * index


def component_means() -> Tuple[np.ndarray, np.ndarray]:
    mean_1 = np.zeros(total_dim, dtype=np.float64)
    mean_2 = np.zeros(total_dim, dtype=np.float64)
    for indices, separation in (
        ([0, 1, 2, 3, 4], 1.35),
        ([5, 6, 7, 8, 9], 0.5),
        ([10, 11, 12, 13, 14], 0.2),
        ([15, 16, 17, 18, 19], 0.1),
    ):
        mean_1[indices] = -separation
        mean_2[indices] = separation
    return mean_1, mean_2


def sample_student_t_mixture(
    means: Iterable[np.ndarray],
    sample_count: int,
) -> np.ndarray:
    means = [np.asarray(mean, dtype=np.float64) for mean in means]
    labels = np.random.choice(
        len(means),
        size=sample_count,
        p=mixture_weights,
    )
    samples = np.empty((sample_count, total_dim), dtype=np.float64)
    for component, mean in enumerate(means):
        indices = np.where(labels == component)[0]
        if indices.size == 0:
            continue
        normal = np.random.randn(indices.size, total_dim)
        chi_square = np.random.chisquare(df=nu_t, size=indices.size)
        scale = student_t_scale / np.sqrt(chi_square / nu_t)
        samples[indices] = mean[None, :] + normal * scale[:, None]
    return samples


def make_training_data(
    dim_u: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    mean_1, mean_2 = component_means()
    samples = sample_student_t_mixture(
        (mean_1, mean_2),
        n_joint_samples,
    )
    samples = samples[np.random.permutation(n_joint_samples)]
    samples = torch.tensor(samples, device=device, dtype=torch.float32)

    u_physical = samples[:, :dim_u]
    v_physical = samples[:, dim_u:]
    mean_u = u_physical.mean(dim=0)
    std_u = u_physical.std(dim=0).clamp_min(1e-8)
    mean_v = v_physical.mean(dim=0)
    std_v = v_physical.std(dim=0).clamp_min(1e-8)
    u_standardized = (u_physical - mean_u) / std_u
    v_standardized = (v_physical - mean_v) / std_v

    nearest = torch.topk(
        torch.linalg.norm(v_physical, dim=1),
        k=n_training_samples,
        largest=False,
    ).indices
    return {
        "u": u_standardized[nearest],
        "v": v_standardized[nearest],
        "mean_u": mean_u,
        "std_u": std_u,
        "mean_v": mean_v,
        "std_v": std_v,
    }


class FourierSigmaEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.linspace(0.1, 100.0, sigma_embedding_frequencies)[None],
        )
        self.phase = nn.Parameter(torch.randn(1, sigma_embedding_frequencies))
        self.network = nn.Sequential(
            nn.Linear(2 * sigma_embedding_frequencies, sigma_embedding_hidden),
            nn.GELU(),
            nn.Linear(sigma_embedding_hidden, sigma_embedding_dim),
            nn.GELU(),
        )

    def forward(self, sigma_value: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(sigma_value[:, None] + 1e-12)
        angle = log_sigma * self.frequencies + self.phase
        return self.network(torch.cat((torch.sin(angle), torch.cos(angle)), dim=1))


class ConditionalScoreNetwork(nn.Module):
    def __init__(self, dim_u: int, dim_v: int) -> None:
        super().__init__()
        self.dim_u = dim_u
        self.sigma_embedding = FourierSigmaEmbedding()
        layers = [
            nn.Linear(dim_u + dim_v + sigma_embedding_dim, width),
            nn.ReLU(),
        ]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, dim_u))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        u_noisy: torch.Tensor,
        v: torch.Tensor,
        sigma_value: torch.Tensor,
    ) -> torch.Tensor:
        embedded_sigma = self.sigma_embedding(sigma_value)
        return self.network(torch.cat((u_noisy, v, embedded_sigma), dim=1))


def sigma_schedule(device: torch.device) -> torch.Tensor:
    return torch.exp(
        torch.linspace(
            math.log(sigma_max),
            math.log(sigma_min),
            noise_levels,
            device=device,
        )
    )


def update_ema(
    ema_state: Dict[str, torch.Tensor],
    model: nn.Module,
) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name not in ema_state:
                ema_state[name] = value.detach().clone()
            else:
                ema_state[name].mul_(ema_decay).add_(
                    value.detach(),
                    alpha=1.0 - ema_decay,
                )


def load_or_train_score_network(
    training: Dict[str, torch.Tensor],
    dim_u: int,
    dim_v: int,
    device: torch.device,
    checkpoint_path: str,
) -> ConditionalScoreNetwork:
    model = ConditionalScoreNetwork(dim_u, dim_v).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["dim_u"] != dim_u or checkpoint["dim_v"] != dim_v:
            raise ValueError("The saved score-network dimensions do not match this run")
        model.load_state_dict(checkpoint["ema_state"])
        model.eval()
        print(f"loaded {checkpoint_path}", flush=True)
        return model

    set_seed(seed)
    model = ConditionalScoreNetwork(dim_u, dim_v).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    steps_per_epoch = math.ceil(n_training_samples / batch_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * steps_per_epoch,
    )
    sigmas = sigma_schedule(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    ema_state: Dict[str, torch.Tensor] = {}

    model.train()
    global_step = 0
    for epoch in range(epochs):
        permutation = torch.randperm(
            n_training_samples,
            device=device,
            generator=generator,
        )
        for start in range(0, n_training_samples, batch_size):
            indices = permutation[start : start + batch_size]
            u = training["u"][indices]
            v = training["v"][indices]
            uniform = torch.rand(indices.numel(), device=device, generator=generator)
            sigma_indices = torch.clamp(
                ((1.0 - uniform.pow(sigma_sampling_power)) * noise_levels).long(),
                0,
                noise_levels - 1,
            )
            selected_sigmas = sigmas[sigma_indices]
            noise = torch.randn(
                (indices.numel(), dim_u),
                device=device,
                generator=generator,
                dtype=torch.float32,
            )
            noisy_u = u + selected_sigmas[:, None] * noise
            prediction = model(noisy_u, v, selected_sigmas)
            residual = selected_sigmas[:, None] * prediction + noise
            loss = 0.5 * residual.square().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()
            update_ema(ema_state, model)

            if global_step % log_every == 0:
                print(
                    f"epoch {epoch}/{epochs}, step {global_step}, "
                    f"loss {loss.item():.6e}",
                    flush=True,
                )
            global_step += 1

    torch.save(
        {
            "dim_u": dim_u,
            "dim_v": dim_v,
            "state_dict": model.state_dict(),
            "ema_state": ema_state,
        },
        checkpoint_path,
    )
    model.load_state_dict(ema_state)
    model.eval()
    return model


def clip_by_norm(values: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = torch.linalg.norm(values, dim=1, keepdim=True)
    scale = torch.clamp(maximum / (norms + 1e-12), max=1.0)
    return values * scale


@torch.no_grad()
def sample_with_ald(
    model: ConditionalScoreNetwork,
    condition_standardized: torch.Tensor,
    sample_count: int,
    device: torch.device,
    sample_seed: int,
) -> torch.Tensor:
    sigmas = sigma_schedule(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(sample_seed)
    samples = torch.randn(
        (sample_count, model.dim_u),
        device=device,
        generator=generator,
    )
    repeated_condition = condition_standardized[None, :].repeat(sample_count, 1)

    for sigma_value in sigmas:
        alpha = langevin_step_size * float(
            sigma_value.square() / sigmas[-1].square()
        )
        backoffs = 0
        for _ in range(langevin_steps_per_level):
            while True:
                noise = torch.randn(
                    samples.shape,
                    device=device,
                    generator=generator,
                )
                score = clip_by_norm(
                    model(
                        samples,
                        repeated_condition,
                        sigma_value.repeat(sample_count),
                    ),
                    score_clip,
                )
                candidate = samples + alpha * score + math.sqrt(2.0 * alpha) * noise
                candidate = torch.clamp(candidate, -sample_clip, sample_clip)
                if torch.isfinite(candidate).all():
                    samples = candidate
                    break
                backoffs += 1
                if backoffs > max_backoffs:
                    invalid = ~torch.isfinite(candidate).all(dim=1)
                    if invalid.any():
                        samples[invalid] = torch.randn(
                            (int(invalid.sum()), model.dim_u),
                            device=device,
                            generator=generator,
                        )
                    samples = torch.nan_to_num(samples)
                    break
                alpha *= 0.5

    final_sigma = sigmas[-1]
    final_score = clip_by_norm(
        model(
            samples,
            repeated_condition,
            final_sigma.repeat(sample_count),
        ),
        score_clip,
    )
    samples = samples + final_sigma.square() * final_score
    return torch.nan_to_num(
        torch.clamp(samples, -sample_clip, sample_clip)
    )


def multivariate_student_t_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
) -> float:
    dimension = value.size
    delta = np.sum((value - mean) ** 2) / student_t_scale**2
    return float(
        gammaln((nu_t + dimension) / 2.0)
        - gammaln(nu_t / 2.0)
        - 0.5 * dimension * np.log(nu_t * np.pi)
        - dimension * np.log(student_t_scale)
        - 0.5 * (nu_t + dimension) * np.log(1.0 + delta / nu_t)
    )


def conditional_parameters(
    condition: float,
    dim_u: int,
    dim_v: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean_1, mean_2 = component_means()
    u_means = np.stack((mean_1[:dim_u], mean_2[:dim_u]))
    v_means = (mean_1[dim_u:], mean_2[dim_u:])
    y = condition * np.ones(dim_v, dtype=np.float64)
    log_weights = np.array(
        [
            np.log(mixture_weights[k] + 1e-300)
            + multivariate_student_t_logpdf(y, v_means[k])
            for k in range(2)
        ]
    )
    log_weights -= log_weights.max()
    posterior_weights = np.exp(log_weights)
    posterior_weights /= posterior_weights.sum()
    deltas = np.array(
        [np.sum((y - v_means[k]) ** 2) / student_t_scale**2 for k in range(2)]
    )
    scales = np.sqrt(
        student_t_scale**2 * (nu_t + deltas) / (nu_t + dim_v)
    )
    return posterior_weights, nu_t + dim_v, u_means, scales


def projection_kl(
    samples: np.ndarray,
    condition: float,
    dim_u: int,
    dim_v: int,
) -> float:
    weights, conditional_nu, locations, scales = conditional_parameters(
        condition,
        dim_u,
        dim_v,
    )
    direction = locations[1] - locations[0]
    direction /= np.linalg.norm(direction) + 1e-14
    projected_samples = samples @ direction
    projected_locations = locations @ direction

    exact_density = np.zeros_like(projection_grid)
    for component in range(2):
        exact_density += (
            weights[component]
            * student_t.pdf(
                (projection_grid - projected_locations[component])
                / scales[component],
                df=conditional_nu,
            )
            / scales[component]
        )
    exact_density = normalize_density(exact_density, projection_grid)
    estimated_density = normalize_density(
        gaussian_kde(projected_samples)(projection_grid),
        projection_grid,
    )
    return kl_divergence(exact_density, estimated_density, projection_grid)


def average_marginal_kl(
    samples: np.ndarray,
    condition: float,
    dim_u: int,
    dim_v: int,
) -> float:
    weights, conditional_nu, locations, scales = conditional_parameters(
        condition,
        dim_u,
        dim_v,
    )
    marginal_kls = []
    for coordinate in range(dim_u):
        exact_density = np.zeros_like(marginal_grid)
        for component in range(2):
            exact_density += (
                weights[component]
                * student_t.pdf(
                    (marginal_grid - locations[component, coordinate])
                    / scales[component],
                    df=conditional_nu,
                )
                / scales[component]
            )
        exact_density = normalize_density(exact_density, marginal_grid)
        estimated_density = normalize_density(
            gaussian_kde(samples[:, coordinate])(marginal_grid),
            marginal_grid,
        )
        marginal_kls.append(
            kl_divergence(
                exact_density,
                estimated_density,
                marginal_grid,
            )
        )
    return float(np.mean(marginal_kls))


def evaluate(
    model: ConditionalScoreNetwork,
    training: Dict[str, torch.Tensor],
    dim_u: int,
    dim_v: int,
    device: torch.device,
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    mean_u = training["mean_u"].detach().cpu().numpy()
    std_u = training["std_u"].detach().cpu().numpy()
    mean_v = training["mean_v"].detach().cpu().numpy()
    std_v = training["std_v"].detach().cpu().numpy()
    for condition in conditions:
        physical_condition = condition * np.ones(dim_v, dtype=np.float32)
        standardized_condition = (physical_condition - mean_v) / std_v
        standardized_samples = sample_with_ald(
            model,
            torch.tensor(
                standardized_condition,
                device=device,
                dtype=torch.float32,
            ),
            n_test_samples,
            device,
            condition_seed(condition),
        ).cpu().numpy()
        physical_samples = standardized_samples * std_u + mean_u
        finite_samples = physical_samples[np.isfinite(physical_samples).all(axis=1)]
        if finite_samples.shape[0] < 500:
            raise RuntimeError(f"Too few finite samples for condition {condition:+.2f}")
        results[condition] = {
            "projection_kl": projection_kl(
                finite_samples,
                condition,
                dim_u,
                dim_v,
            ),
            "average_marginal_kl": average_marginal_kl(
                finite_samples,
                condition,
                dim_u,
                dim_v,
            ),
        }
        print(
            f"condition {condition:+.2f}, seed {condition_seed(condition)}: "
            f"projection KL = {results[condition]['projection_kl']:.10f}, "
            "average marginal KL = "
            f"{results[condition]['average_marginal_kl']:.10f}",
            flush=True,
        )
    return results


def write_results(
    results: Dict[float, Dict[str, float]],
    output_dir: str,
) -> None:
    for filename, metric in (
        ("projection_kl.csv", "projection_kl"),
        ("average_marginal_kl.csv", "average_marginal_kl"),
    ):
        csv_path = os.path.join(output_dir, filename)
        with open(csv_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["method", *[f"y={condition:g}" for condition in table_conditions]]
            )
            writer.writerow(
                [
                    "Score Network",
                    *[
                        f"{results[condition][metric]:.10f}"
                        for condition in table_conditions
                    ],
                ]
            )
        print(f"saved {csv_path}", flush=True)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dim_u = args.dim_u
    dim_v = total_dim - dim_u
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            "student_t_score_network_results",
            f"dim{dim_u}_{dim_v}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "score_network.pt")

    print(f"device: {device}", flush=True)
    print(f"dimensions: {dim_u}/{dim_v}", flush=True)
    print(f"seed: {seed}", flush=True)
    training = make_training_data(dim_u, device)
    model = load_or_train_score_network(
        training,
        dim_u,
        dim_v,
        device,
        checkpoint_path,
    )
    results = evaluate(model, training, dim_u, dim_v, device)
    write_results(results, output_dir)


if __name__ == "__main__":
    main()
