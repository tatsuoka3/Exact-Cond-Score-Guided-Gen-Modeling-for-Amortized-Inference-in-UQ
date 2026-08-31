#!/usr/bin/env python3
"""From-scratch Student-t mixture NN/DM experiment.

Run with ``--dim-u 15`` or ``--dim-u 10``.  This program intentionally does
not load any old checkpoint, normalization array, training label, or cached
sample.  It regenerates the data and diffusion labels, trains a new NN, and
then evaluates both the new NN and the DM at all three fixed conditions.

The data are standardized as Uz=(U-mean_U)/std_U and
Vz=(V-mean_V)/std_V. Physical scalar diffusion variances supplied on the
command line are converted coordinatewise via
VAR_Uz[j]=VAR_U_physical/std_U[j]**2 and analogously for V and Y.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gammaln
from scipy.stats import gaussian_kde, t as student_t

from shared_experiment_utils import (
    b,
    conditional_score,
    kl_divergence,
    normalize_density,
    reverse_sde,
    sigma,
    standardize_variance,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234
TOTAL_DIM = 20
N_SAMPLE = 250_000
N_GEN = 50_000
N_TEST = 5_000
CONDITIONS = (-0.25, 0.0, 0.25)

NU_T = 10.0
SIGMA_VALUE = 1.0
WEIGHTS = np.array([0.5, 0.5], dtype=float)

DEFAULT_VAR_U_PHYSICAL = 0.3
DEFAULT_VAR_V_PHYSICAL = 0.1
DEFAULT_VAR_Y_PHYSICAL = 1e-5

DM_STEPS = 100
# Computational microbatch only.  Batch 150 matches the requested/reference
# experiment.  It requires a GPU with more than 17.4 GiB available for the
# measured dim_u=10, dim_v=10 case.
DEFAULT_DM_BATCH = 150

ACT = "relu"
DEPTH = 6
WIDTH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SCHEDULER = "cosine"
EPOCHS = 50_000
BATCH_SIZE = 4_096
PRINT_EVERY = 200

PROJ_GRID = {"min": -10.0, "max": 10.0, "n": 500}
# Exact marginal reporting grid from the supplied KL-reporting script.
MARG_GRID = {"min": -8.0, "max": 8.0, "n": 1200}

TAG = (
    "plain_actrelu_depth6_width128_lr0.001_wd0.0_"
    "schedcosine_epochs50000_bs4096_2"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Student-t mixture NN/DM setting using physical "
            "diffusion variances converted coordinatewise after standardization."
        )
    )
    parser.add_argument("--dim-u", type=int, choices=(10, 15), required=True)
    parser.add_argument(
        "--var-u-physical",
        type=float,
        default=DEFAULT_VAR_U_PHYSICAL,
    )
    parser.add_argument(
        "--var-v-physical",
        type=float,
        default=DEFAULT_VAR_V_PHYSICAL,
    )
    parser.add_argument(
        "--var-y-physical",
        type=float,
        default=DEFAULT_VAR_Y_PHYSICAL,
    )
    parser.add_argument(
        "--dm-batch",
        type=int,
        default=DEFAULT_DM_BATCH,
        help=(
            "Reverse-diffusion computational microbatch. Default: 150."
        ),
    )
    parser.add_argument("--home-root", default=".")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer(name: str, timing_store: Dict[str, float]):
    sync()
    start = time.time()
    try:
        yield
    finally:
        sync()
        elapsed = time.time() - start
        timing_store[name] = float(elapsed)
        print(f"[TIMER] {name}: {elapsed:.6f} s", flush=True)


def seed_for_condition(index: int) -> int:
    return int(SEED + 10_007 * index)


def condition_key(condition: float) -> str:
    return str(float(condition))


def condition_label(condition: float) -> str:
    return f"{float(condition):+.2f}"


def write_json(payload: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def sample_mvt_isotropic(
    mu: np.ndarray,
    sigma_value: float,
    nu: float,
    sample_count: int,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    z = np.random.randn(sample_count, mu.size)
    s = np.random.chisquare(df=nu, size=sample_count)
    scale = sigma_value / np.sqrt(s / nu)
    return mu[None, :] + z * scale[:, None]


def sample_two_component_student_t(
    means: Iterable[np.ndarray],
    sigma_value: float,
    nu: float,
    weights: np.ndarray,
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights /= normalized_weights.sum()
    means_list = [np.asarray(mean, dtype=np.float64).reshape(-1) for mean in means]
    labels = np.random.choice(len(means_list), size=sample_count, p=normalized_weights)
    samples = np.empty((sample_count, means_list[0].size), dtype=np.float64)
    for component, mean in enumerate(means_list):
        indices = np.where(labels == component)[0]
        if indices.size:
            samples[indices] = sample_mvt_isotropic(
                mean,
                sigma_value,
                nu,
                indices.size,
            )
    return samples, labels


def component_means() -> Tuple[np.ndarray, np.ndarray]:
    mu1 = np.zeros(TOTAL_DIM, dtype=np.float64)
    mu2 = np.zeros(TOTAL_DIM, dtype=np.float64)
    for indices, delta in (
        ([0, 1, 2, 3, 4], 1.35),
        ([5, 6, 7, 8, 9], 0.5),
        ([10, 11, 12, 13, 14], 0.2),
        ([15, 16, 17, 18, 19], 0.1),
    ):
        mu1[indices] = -delta
        mu2[indices] = delta
    return mu1, mu2


def closest_conditions(v_phys: torch.Tensor, count: int) -> torch.Tensor:
    target = torch.zeros(v_phys.shape[1], device=v_phys.device, dtype=v_phys.dtype)
    distances = torch.norm(v_phys - target[None, :], dim=1)
    indices = torch.topk(distances, k=count, largest=False).indices
    return v_phys[indices]


def generate_training_data(
    dim_u: int,
    dim_v: int,
    mu1: np.ndarray,
    mu2: np.ndarray,
    output_dir: str,
    dm_batch: int,
    var_u_physical: float,
    var_v_physical: float,
    var_y_physical: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, str]]:
    timings: Dict[str, float] = {}
    files: Dict[str, str] = {}
    total_start = time.time()

    with timer("sample_joint_physical", timings):
        x_phys, labels = sample_two_component_student_t(
            [mu1, mu2],
            SIGMA_VALUE,
            NU_T,
            WEIGHTS,
            N_SAMPLE,
        )
    permutation = np.random.permutation(N_SAMPLE)
    x_phys = x_phys[permutation].astype(np.float64)
    labels = labels[permutation]
    x_phys_tensor = torch.tensor(x_phys, device=DEVICE, dtype=torch.float32)
    u_phys = x_phys_tensor[:, :dim_u]
    v_phys = x_phys_tensor[:, dim_u:]

    with timer("standardize_U_V", timings):
        mean_u = u_phys.mean(dim=0)
        std_u = u_phys.std(dim=0).clamp_min(1e-8)
        mean_v = v_phys.mean(dim=0)
        std_v = v_phys.std(dim=0).clamp_min(1e-8)
        u_normalized = (u_phys - mean_u) / std_u
        v_normalized = (v_phys - mean_v) / std_v

    arrays_to_save = {
        "mean_U": mean_u,
        "std_U": std_u,
        "mean_V": mean_v,
        "std_V": std_v,
        "sample_U_normalized": u_normalized,
        "sample_V_normalized": v_normalized,
    }
    for name, tensor in arrays_to_save.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, tensor.detach().cpu().numpy())
        files[name] = path
    labels_path = os.path.join(output_dir, "joint_component_labels.npy")
    np.save(labels_path, labels)
    files["joint_component_labels"] = labels_path

    with timer("select_training_conditions", timings):
        cond_y_phys = closest_conditions(v_phys, N_GEN)
        cond_y = (cond_y_phys - mean_v[None, :]) / std_v[None, :]
    with timer("sample_training_xT", timings):
        x_terminal = torch.randn(N_GEN, TOTAL_DIM, device=DEVICE, dtype=torch.float32)

    # Preserve the requested physical variances under the coordinate change.
    # Because standardization is coordinatewise, the converted variances are
    # per-dimensional vectors rather than fixed scalars.
    var_u_normalized = standardize_variance(var_u_physical, std_u)
    var_v_normalized = standardize_variance(var_v_physical, std_v)
    var_y_normalized = standardize_variance(var_y_physical, std_v)

    generated_batches = []
    batch_count = (N_GEN + dm_batch - 1) // dm_batch
    with timer("reverse_diffusion_training_labels", timings):
        with torch.no_grad():
            for batch_index in range(batch_count):
                start = batch_index * dm_batch
                stop = min((batch_index + 1) * dm_batch, N_GEN)
                score_function = partial(
                    conditional_score,
                    sample_u=u_normalized,
                    sample_v=v_normalized,
                    condition_y=cond_y[start:stop],
                    variance_u=var_u_normalized,
                    variance_v=var_v_normalized,
                    variance_y=var_y_normalized,
                )
                generated = reverse_sde(
                    x_terminal=x_terminal[start:stop],
                    time_steps=DM_STEPS,
                    drift=b,
                    diffusion=sigma,
                    score=score_function,
                    save_path=False,
                )
                generated_batches.append(generated)
                if (batch_index + 1) % 10 == 0 or (batch_index + 1) == batch_count:
                    print(
                        f"  training-label DM batch {batch_index + 1}/{batch_count}",
                        flush=True,
                    )
    regenerated_normalized = torch.cat(generated_batches, dim=0)

    for name, tensor in {
        "cond_Y_normalized": cond_y,
        "xT_amortized": x_terminal,
        "samples_regen_normalized": regenerated_normalized,
    }.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, tensor.detach().cpu().numpy())
        files[name] = path

    timings["total_data_labeling"] = float(time.time() - total_start)
    return (
        {
            "mean_u": mean_u,
            "std_u": std_u,
            "mean_v": mean_v,
            "std_v": std_v,
            "u_normalized": u_normalized,
            "v_normalized": v_normalized,
            "cond_y": cond_y,
            "x_terminal": x_terminal,
            "regenerated_normalized": regenerated_normalized,
            "var_u_normalized": var_u_normalized,
            "var_v_normalized": var_v_normalized,
            "var_y_normalized": var_y_normalized,
        },
        timings,
        files,
    )


def activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(name)


class PlainMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        act = activation(ACT)
        layers = [nn.Linear(input_dim, WIDTH), act]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(WIDTH, WIDTH), act]
        layers += [nn.Linear(WIDTH, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_network(
    dim_u: int,
    dim_v: int,
    training: Mapping[str, torch.Tensor],
    output_dir: str,
) -> Tuple[nn.Module, Dict[str, Any], float, str]:
    network_input = torch.hstack(
        (
            training["cond_y"].reshape(-1, dim_v),
            training["x_terminal"].reshape(-1, TOTAL_DIM),
        )
    )
    target = training["regenerated_normalized"][:, :dim_u].reshape(-1, dim_u)
    model = PlainMLP(network_input.shape[1], target.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_function = nn.MSELoss()
    sample_count = network_input.shape[0]
    batch_count = (sample_count + BATCH_SIZE - 1) // BATCH_SIZE
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = np.empty(EPOCHS, dtype=np.float64)

    sync()
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(sample_count, device=DEVICE)
        shuffled_input = network_input[permutation]
        shuffled_target = target[permutation]
        running = 0.0
        for batch_index in range(batch_count):
            start = batch_index * BATCH_SIZE
            stop = min((batch_index + 1) * BATCH_SIZE, sample_count)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(shuffled_input[start:stop])
            loss = loss_function(prediction, shuffled_target[start:stop])
            loss.backward()
            optimizer.step()
            running += loss.item() * (stop - start)
        epoch_loss = running / sample_count
        history[epoch] = epoch_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()
        if epoch % PRINT_EVERY == 0:
            print(
                f"  epoch {epoch:6d} mse={epoch_loss:.8e} "
                f"best={best_loss:.8e} lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
    sync()
    elapsed = float(time.time() - start_time)
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    record = {
        "tag": TAG,
        "best_train_mse": float(best_loss),
        "best_epoch": int(best_epoch),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_hp": {"width": WIDTH, "depth": DEPTH, "act": ACT},
        "optim_hp": {
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER,
        },
        "device": DEVICE,
        "note": (
            "Retrained from scratch with seed 1234; physical diffusion variances "
            "converted coordinatewise by empirical std**2."
        ),
    }
    checkpoint_path = os.path.join(output_dir, f"FN_MSE_{TAG}.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": int(network_input.shape[1]),
            "out_dim": int(target.shape[1]),
            "rec": record,
        },
        checkpoint_path,
    )
    np.save(os.path.join(output_dir, "nn_training_loss_by_epoch.npy"), history)
    print(f"[TIMER] total_NN_training: {elapsed:.6f} s", flush=True)
    print(f"[saved] {checkpoint_path}", flush=True)
    return model, record, elapsed, checkpoint_path


def mvt_logpdf(x: np.ndarray, mu: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    dimension = x.size
    delta = np.sum((x - mu) ** 2) / SIGMA_VALUE**2
    return float(
        gammaln((NU_T + dimension) / 2.0)
        - gammaln(NU_T / 2.0)
        - 0.5 * dimension * np.log(NU_T * np.pi)
        - dimension * np.log(SIGMA_VALUE)
        - 0.5 * (NU_T + dimension) * np.log(1.0 + delta / NU_T)
    )


def conditional_parameters(
    condition: float,
    dim_u: int,
    dim_v: int,
    mu1: np.ndarray,
    mu2: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    y = float(condition) * np.ones(dim_v, dtype=float)
    mu_u = [mu1[:dim_u], mu2[:dim_u]]
    mu_v = [mu1[dim_u:], mu2[dim_u:]]
    log_weights = np.array(
        [np.log(WEIGHTS[k] + 1e-300) + mvt_logpdf(y, mu_v[k]) for k in range(2)]
    )
    log_weights -= np.max(log_weights)
    posterior_weights = np.exp(log_weights)
    posterior_weights /= posterior_weights.sum()
    deltas = np.array(
        [np.sum((y - mu_v[k]) ** 2) / SIGMA_VALUE**2 for k in range(2)]
    )
    scales = np.sqrt(SIGMA_VALUE**2 * (NU_T + deltas) / (NU_T + dim_v))
    return posterior_weights, NU_T + dim_v, np.stack(mu_u), scales


def true_1d_pdf(
    grid: np.ndarray,
    weights: np.ndarray,
    nu: float,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    density = np.zeros_like(grid)
    for component in range(2):
        density += (
            weights[component]
            * student_t.pdf(
                (grid - locations[component]) / scales[component],
                df=nu,
            )
            / scales[component]
        )
    return density


def kl_metrics(
    samples: np.ndarray,
    condition: float,
    dim_u: int,
    dim_v: int,
    mu1: np.ndarray,
    mu2: np.ndarray,
) -> Tuple[float, float]:
    weights, conditional_nu, locations, scales = conditional_parameters(
        condition,
        dim_u,
        dim_v,
        mu1,
        mu2,
    )
    direction = locations[1] - locations[0]
    direction /= np.linalg.norm(direction) + 1e-14
    projected = samples @ direction
    grid = np.linspace(PROJ_GRID["min"], PROJ_GRID["max"], PROJ_GRID["n"])
    projected_locations = np.array([direction @ locations[0], direction @ locations[1]])
    p = normalize_density(
        true_1d_pdf(grid, weights, conditional_nu, projected_locations, scales),
        grid,
    )
    q = normalize_density(
        gaussian_kde(projected)(grid),
        grid,
    )
    projection_kl = kl_divergence(p, q, grid)

    marginal_grid = np.linspace(MARG_GRID["min"], MARG_GRID["max"], MARG_GRID["n"])
    marginal_total = 0.0
    for coordinate in range(dim_u):
        coordinate_locations = locations[:, coordinate]
        p_coordinate = normalize_density(
            true_1d_pdf(
                marginal_grid,
                weights,
                conditional_nu,
                coordinate_locations,
                scales,
            ),
            marginal_grid,
        )
        q_coordinate = normalize_density(
            gaussian_kde(samples[:, coordinate])(marginal_grid),
            marginal_grid,
        )
        marginal_total += kl_divergence(
            p_coordinate,
            q_coordinate,
            marginal_grid,
        )
    return float(projection_kl), float(marginal_total / dim_u)


@torch.no_grad()
def generate_nn_samples(
    model: nn.Module,
    training: Mapping[str, torch.Tensor],
    dim_u: int,
    dim_v: int,
    condition: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_phys = float(condition) * np.ones((N_TEST, dim_v), dtype=np.float32)
    mean_v = training["mean_v"].detach().cpu().numpy()
    std_v = training["std_v"].detach().cpu().numpy()
    y_normalized = (y_phys - mean_v[None, :]) / std_v[None, :]
    z_terminal = rng.standard_normal((N_TEST, TOTAL_DIM)).astype(np.float32)
    network_input = np.hstack([y_normalized, z_terminal]).astype(np.float32)
    u_normalized = model(
        torch.tensor(network_input, device=DEVICE, dtype=torch.float32)
    ).detach().cpu().numpy()
    mean_u = training["mean_u"].detach().cpu().numpy()
    std_u = training["std_u"].detach().cpu().numpy()
    u_phys = u_normalized * std_u[None, :] + mean_u[None, :]
    return u_phys.astype(np.float64), z_terminal


@torch.no_grad()
def generate_dm_samples(
    training: Mapping[str, torch.Tensor],
    dim_u: int,
    dim_v: int,
    condition: float,
    seed: int,
    dm_batch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    y_phys = float(condition) * np.ones((1, dim_v), dtype=np.float32)
    mean_v_np = training["mean_v"].detach().cpu().numpy()
    std_v_np = training["std_v"].detach().cpu().numpy()
    y_normalized = (y_phys - mean_v_np[None, :]) / std_v_np[None, :]
    cond_y = torch.tensor(y_normalized, device=DEVICE, dtype=torch.float32).repeat(N_TEST, 1)
    x_terminal = torch.randn(N_TEST, TOTAL_DIM, device=DEVICE, dtype=torch.float32)
    generated_batches = []
    batch_count = (N_TEST + dm_batch - 1) // dm_batch
    for batch_index in range(batch_count):
        start = batch_index * dm_batch
        stop = min((batch_index + 1) * dm_batch, N_TEST)
        score_function = partial(
            conditional_score,
            sample_u=training["u_normalized"],
            sample_v=training["v_normalized"],
            condition_y=cond_y[start:stop],
            variance_u=training["var_u_normalized"],
            variance_v=training["var_v_normalized"],
            variance_y=training["var_y_normalized"],
        )
        generated_batches.append(
            reverse_sde(
                x_terminal=x_terminal[start:stop],
                time_steps=DM_STEPS,
                drift=b,
                diffusion=sigma,
                score=score_function,
                save_path=False,
            )
        )
        if (batch_index + 1) % 5 == 0 or (batch_index + 1) == batch_count:
            print(f"  evaluation DM batch {batch_index + 1}/{batch_count}", flush=True)
    generated_normalized = torch.cat(generated_batches, dim=0)
    u_normalized = generated_normalized[:, :dim_u]
    u_phys = (
        u_normalized * training["std_u"][None, :] + training["mean_u"][None, :]
    )
    return (
        u_phys.detach().cpu().numpy().astype(np.float64),
        generated_normalized.detach().cpu().numpy().astype(np.float32),
        x_terminal.detach().cpu().numpy().astype(np.float32),
    )


def save_condition_samples(
    output_dir: str,
    condition: float,
    seed: int,
    u_nn: np.ndarray,
    z_nn: np.ndarray,
    u_dm: np.ndarray,
    x_dm_normalized: np.ndarray,
    z_dm: np.ndarray,
) -> Dict[str, str]:
    suffix = f"cond{condition_label(condition)}_N{N_TEST}"
    payload = {
        "U_nn_phys": u_nn,
        "zT_nn": z_nn,
        "U_dm_phys": u_dm,
        "X_dm_normalized": x_dm_normalized,
        "zT_dm": z_dm,
    }
    paths: Dict[str, str] = {}
    for name, array in payload.items():
        path = os.path.join(output_dir, f"{name}_{suffix}.npy")
        np.save(path, array)
        paths[name] = path
    return paths


def write_csv(results: Mapping[str, Any], path: str) -> None:
    fields = [
        "condition",
        "seed",
        "nn_generation_s",
        "dm_generation_s",
        "nn_proj_KL",
        "nn_avgKL",
        "dm_proj_KL",
        "dm_avgKL",
    ]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            entry = results["evaluation"][condition_key(condition)]
            writer.writerow(
                {
                    "condition": condition,
                    "seed": entry["seed"],
                    "nn_generation_s": entry["timing_s"]["NN_generation"],
                    "dm_generation_s": entry["timing_s"]["DM_generation"],
                    "nn_proj_KL": entry["KL"]["NN"]["proj_KL"],
                    "nn_avgKL": entry["KL"]["NN"]["avgKL"],
                    "dm_proj_KL": entry["KL"]["DM"]["proj_KL"],
                    "dm_avgKL": entry["KL"]["DM"]["avgKL"],
                }
            )


def main() -> None:
    args = parse_args()
    if DEVICE != "cuda":
        raise RuntimeError("CUDA is required for this from-scratch reproduction.")
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    dim_u = int(args.dim_u)
    dim_v = TOTAL_DIM - dim_u
    dm_batch = int(args.dm_batch)
    var_u_physical = float(args.var_u_physical)
    var_v_physical = float(args.var_v_physical)
    var_y_physical = float(args.var_y_physical)
    if dm_batch <= 0:
        raise ValueError(f"--dm-batch must be positive; received {dm_batch}.")
    for name, value in (
        ("--var-u-physical", var_u_physical),
        ("--var-v-physical", var_v_physical),
        ("--var-y-physical", var_y_physical),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; received {value}.")
    job_token = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    variance_token = (
        f"VAR_U_{var_u_physical:g}_VAR_V_{var_v_physical:g}_"
        f"VAR_Y_{var_y_physical:g}"
    )
    output_dir = os.path.abspath(
        args.output_dir
        if args.output_dir is not None
        else os.path.join(
            args.home_root,
            "student_t_mixture_results",
            variance_token,
            f"dim_u_{dim_u}_dim_v_{dim_v}",
            f"job_{job_token}",
        )
    )
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to reuse nonempty output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "student_t_mixture_results.json")
    csv_path = os.path.join(output_dir, "student_t_mixture_timing_and_KL.csv")
    text_path = os.path.join(output_dir, "student_t_mixture_summary.txt")
    mu1, mu2 = component_means()
    total_start = time.time()

    results: Dict[str, Any] = {
        "status": "running",
        "description": (
            "Student-t mixture data generation, DM labeling, NN training, "
            "and NN/DM evaluation using physical variances converted "
            "coordinatewise after standardization."
        ),
        "loaded_old_artifacts": False,
        "device": DEVICE,
        "seed": SEED,
        "dimensions": {"dim_u": dim_u, "dim_v": dim_v, "D": TOTAL_DIM},
        "sizes": {"N_sample": N_SAMPLE, "N_gen": N_GEN, "N_test": N_TEST},
        "diffusion": {
            "steps": DM_STEPS,
            "batch": dm_batch,
            "batch_role": "computational microbatch only; does not change N_sample/N_gen/N_test",
            "VAR_U_physical": var_u_physical,
            "VAR_V_physical": var_v_physical,
            "VAR_Y_physical": var_y_physical,
            "variance_mode": "physical_to_standardized_coordinatewise",
            "physical_to_standardized_variance_conversion": True,
            "conversion": {
                "VAR_Uz_j": "VAR_U_physical/std_U_j^2",
                "VAR_Vz_j": "VAR_V_physical/std_V_j^2",
                "VAR_Yz_j": "VAR_Y_physical/std_V_j^2",
            },
        },
        "network": {
            "width": WIDTH,
            "depth": DEPTH,
            "activation": ACT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "scheduler": SCHEDULER,
        },
        "conditions": list(CONDITIONS),
        "condition_seeds": {
            condition_key(c): seed_for_condition(i) for i, c in enumerate(CONDITIONS)
        },
        "grids": {"projection": PROJ_GRID, "marginal": MARG_GRID},
        "KL_direction": "D_KL(p_true || p_model_KDE)",
        "KL_reporting": {
            "projection_grid": PROJ_GRID,
            "marginal_grid": MARG_GRID,
            "finite_grid_density_normalization": "trapezoidal",
            "density_clip_floor": 1e-12,
            "KL_sum": "sum(p*log(p/q))*dx",
            "KDE": "scipy.stats.gaussian_kde default Scott bandwidth",
        },
        "output_dir": output_dir,
        "evaluation": {},
    }
    write_json(results, results_path)
    print(f"[setting] U^{dim_u}, V^{dim_v}", flush=True)
    print(f"[output] {output_dir}", flush=True)
    print("[mode] FROM SCRATCH: no old checkpoint or cache is loaded", flush=True)
    print(
        "[variance mode] physical variances divided coordinatewise by empirical std**2",
        flush=True,
    )
    print(
        f"[physical variances] U={var_u_physical:g}, V={var_v_physical:g}, "
        f"Y={var_y_physical:g}",
        flush=True,
    )
    print(f"[DM microbatch] {dm_batch}", flush=True)

    training, data_timing, training_files = generate_training_data(
        dim_u,
        dim_v,
        mu1,
        mu2,
        output_dir,
        dm_batch,
        var_u_physical,
        var_v_physical,
        var_y_physical,
    )
    results["data_labeling_timing_s"] = data_timing
    results["training_data_files"] = training_files
    results["standardized_coordinate_variances_used"] = {
        "VAR_Uz": training["var_u_normalized"].detach().cpu().tolist(),
        "VAR_Vz": training["var_v_normalized"].detach().cpu().tolist(),
        "VAR_Yz": training["var_y_normalized"].detach().cpu().tolist(),
        "VAR_Uz_mean": float(training["var_u_normalized"].mean().item()),
        "VAR_Vz_mean": float(training["var_v_normalized"].mean().item()),
        "VAR_Yz_mean": float(training["var_y_normalized"].mean().item()),
    }
    write_json(results, results_path)

    model, checkpoint_record, training_seconds, checkpoint_path = train_network(
        dim_u,
        dim_v,
        training,
        output_dir,
    )
    results["new_checkpoint"] = checkpoint_path
    results["checkpoint_record"] = checkpoint_record
    results["NN_training_time_s"] = training_seconds
    write_json(results, results_path)

    all_u_nn = []
    all_z_nn = []
    all_u_dm = []
    all_x_dm = []
    all_z_dm = []
    for condition_index, condition in enumerate(CONDITIONS):
        seed = seed_for_condition(condition_index)
        key = condition_key(condition)
        print(f"\n=== condition {condition:+.2f}, seed={seed} ===", flush=True)

        sync()
        start = time.time()
        u_nn, z_nn = generate_nn_samples(
            model,
            training,
            dim_u,
            dim_v,
            condition,
            seed,
        )
        sync()
        nn_seconds = float(time.time() - start)
        nn_proj, nn_avg = kl_metrics(u_nn, condition, dim_u, dim_v, mu1, mu2)

        sync()
        start = time.time()
        u_dm, x_dm_normalized, z_dm = generate_dm_samples(
            training,
            dim_u,
            dim_v,
            condition,
            seed,
            dm_batch,
        )
        sync()
        dm_seconds = float(time.time() - start)
        dm_proj, dm_avg = kl_metrics(u_dm, condition, dim_u, dim_v, mu1, mu2)

        sample_files = save_condition_samples(
            output_dir,
            condition,
            seed,
            u_nn,
            z_nn,
            u_dm,
            x_dm_normalized,
            z_dm,
        )
        results["evaluation"][key] = {
            "condition": condition,
            "seed": seed,
            "timing_s": {"NN_generation": nn_seconds, "DM_generation": dm_seconds},
            "KL": {
                "NN": {"proj_KL": nn_proj, "avgKL": nn_avg},
                "DM": {"proj_KL": dm_proj, "avgKL": dm_avg},
            },
            "sample_files": sample_files,
        }
        all_u_nn.append(u_nn)
        all_z_nn.append(z_nn)
        all_u_dm.append(u_dm)
        all_x_dm.append(x_dm_normalized)
        all_z_dm.append(z_dm)
        write_json(results, results_path)
        print(f"  NN generation: {nn_seconds:.6f} s", flush=True)
        print(f"  DM generation: {dm_seconds:.6f} s", flush=True)
        print(f"  NN KL: proj={nn_proj:.10f}, avg={nn_avg:.10f}", flush=True)
        print(f"  DM KL: proj={dm_proj:.10f}, avg={dm_avg:.10f}", flush=True)
        print(f"  saved: {sample_files}", flush=True)

    results["KL_summary"] = {}
    for method in ("NN", "DM"):
        projected_values = [
            results["evaluation"][condition_key(condition)]["KL"][method]["proj_KL"]
            for condition in CONDITIONS
        ]
        marginal_values = [
            results["evaluation"][condition_key(condition)]["KL"][method]["avgKL"]
            for condition in CONDITIONS
        ]
        results["KL_summary"][method] = {
            "mean_proj_KL": float(np.mean(projected_values)),
            "mean_avgKL": float(np.mean(marginal_values)),
        }

    combined_path = os.path.join(
        output_dir,
        "student_t_mixture_samples_all_conditions.npz",
    )
    np.savez(
        combined_path,
        conditions=np.asarray(CONDITIONS),
        seeds=np.asarray([seed_for_condition(i) for i in range(len(CONDITIONS))]),
        U_nn_phys=np.stack(all_u_nn),
        zT_nn=np.stack(all_z_nn),
        U_dm_phys=np.stack(all_u_dm),
        X_dm_normalized=np.stack(all_x_dm),
        zT_dm=np.stack(all_z_dm),
    )
    results["combined_evaluation_samples"] = combined_path
    results["total_wall_time_s"] = float(time.time() - total_start)
    results["status"] = "completed"
    write_json(results, results_path)
    write_csv(results, csv_path)
    with open(text_path, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(results, indent=2))
        stream.write("\n")
    print(f"\n[saved] {results_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)
    print(f"[saved] {combined_path}", flush=True)
    print(f"[total wall] {results['total_wall_time_s']:.6f} s", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
