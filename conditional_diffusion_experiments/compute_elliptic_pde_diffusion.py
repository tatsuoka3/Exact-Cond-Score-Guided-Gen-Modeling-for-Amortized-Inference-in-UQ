#!/usr/bin/env python3
"""Run conditional diffusion and train the elliptic PDE neural map.

Physical variances are standardized coordinatewise with the shared experiment
utilities. The required PDE dataset is generated separately.
"""

import argparse
import os
import time
import hashlib
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

from shared_experiment_utils import (
    b,
    conditional_score,
    reverse_sde,
    sigma,
    standardize_variance,
)


parser = argparse.ArgumentParser(
    description="Run the elliptic PDE diffusion and neural-map experiment."
)
parser.add_argument(
    "--output-dir",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "elliptic_pde_results",
    ),
    help="Directory containing data/ and receiving experiment results.",
)
args = parser.parse_args()

# ---------------------------
# Device + seed
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

def cuda_sync():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def stable_short_hash(s: str, n=10) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]

# ---------------------------
# Simple MLP you can control
# ---------------------------
class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size=100, n_layers=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hid_size)))
            layers.append(nn.Tanh())
            in_dim = int(hid_size)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ============================================================
# USER PARAMETERS
# ============================================================
K = 20000
N_location_samples = 10
N_coeffs = 4

# DM generation sizes (LABEL DATA GENERATION)
N_COND_TRAIN = 5_000
TIME_STEPS = 1000
DM_BATCH = 1000

# Conditioning noise injected into training V (optional)
ADD_NOISE_TO_TRAIN_V = True
y_std = 0.0025




# ------------------------------------------------------------
# Variance mode
# ------------------------------------------------------------
# Physical variances are converted coordinatewise after standardization.
VAR_MODE = "physical_to_normalized"
# Physical-space scalar knobs
VAR_Y = 1e-6
VAR_U = 1e-2
VAR_V = 1e-5




# FN training knobs
LEARNING_RATE = 1e-3
n_neurons = 128
n_layers = 1
total_epochs = 20000

# NN sampling after training
TESTCASES_TO_TEST = [1, 2]
N_NN_SAMPLES = 5000

# PURE diffusion sampling (post-training)
N_DM_SAMPLES_TEST = 5000
DM_TEST_BATCH = 1000
DM_TEST_TIME_STEPS = TIME_STEPS

# Inference microbenchmark settings (pure NN forward only)
INFER_WARMUP = 10
INFER_REPEATS = 50

# Paths
savedir = os.path.abspath(args.output_dir)
data_dir = os.path.join(savedir, "data")
os.makedirs(savedir, exist_ok=True)

dim_u = N_coeffs
dim_v = N_location_samples
dim_x = dim_u + dim_v

tag = (
    f"elliptic_pde_N{N_COND_TRAIN}_VARMODE_{VAR_MODE}_VAR_Y_{VAR_Y}_VAR_U_{VAR_U}_VAR_V_{VAR_V}_"
    f"yStd_{y_std if ADD_NOISE_TO_TRAIN_V else 0.0}_"
    f"FNw{n_neurons}_FNlayers{n_layers}_K_{K}"
)
tag_hash = stable_short_hash(tag)

timing_csv = os.path.join(savedir, f"TIMING_singlearch_{tag}_{tag_hash}_{K}_K.csv")
variance_json = os.path.join(savedir, f"variance_report_{tag}_{tag_hash}_{K}_K.json")
variance_txt = os.path.join(savedir, f"variance_report_{tag}_{tag_hash}_{K}_K.txt")

print(f"[run] DEVICE={DEVICE}")
print(f"[run] tag={tag}")
print(f"[run] tag_hash={tag_hash}")
print(f"[timing] CSV -> {timing_csv}")

# Initialize timing fields
t_load_prep = float("nan")
t_dm_total = float("nan")
t_dm_per = float("nan")
t_build = float("nan")
t_train_total = float("nan")
t_train_per_epoch = float("nan")
best_epoch = -1
best_loss = float("inf")
t_lossplot = float("nan")
t_infer_avg_batch = float("nan")
t_infer_avg_sample = float("nan")
t_infer_e2e = {}
t_dm_test_total = {}
t_dm_test_per = {}
dm_test_out_path = {}

# ============================================================
# Helper prints for scale diagnostics
# ============================================================
def print_block_scale_stats(name: str, X: torch.Tensor):
    """
    Print scale diagnostics for a block X of shape (N, d)
    in physical space.
    """
    X_cpu = X.detach().cpu()

    mean_dim = X_cpu.mean(dim=0)
    std_dim = X_cpu.std(dim=0)
    min_dim = X_cpu.min(dim=0).values
    max_dim = X_cpu.max(dim=0).values
    mean_abs_dim = X_cpu.abs().mean(dim=0)
    rms_dim = torch.sqrt((X_cpu ** 2).mean(dim=0))

    print(f"\n=== Physical-space scale diagnostics: {name} ===")
    print(f"{name} shape = {tuple(X_cpu.shape)}")
    print(f"{name} per-dim mean:")
    print(mean_dim)
    print(f"{name} per-dim std:")
    print(std_dim)
    print(f"{name} per-dim min:")
    print(min_dim)
    print(f"{name} per-dim max:")
    print(max_dim)
    print(f"{name} per-dim mean(|x|):")
    print(mean_abs_dim)
    print(f"{name} per-dim RMS:")
    print(rms_dim)

    print(f"\n{name} block summaries:")
    print(f"  mean of stds         = {std_dim.mean().item():.6e}")
    print(f"  min  of stds         = {std_dim.min().item():.6e}")
    print(f"  max  of stds         = {std_dim.max().item():.6e}")
    print(f"  mean of variances    = {(std_dim**2).mean().item():.6e}")
    print(f"  mean of mean(|x|)    = {mean_abs_dim.mean().item():.6e}")
    print(f"  mean of RMS          = {rms_dim.mean().item():.6e}")

# ============================================================
# Variance helper
# ============================================================
def build_diffusion_variances(
    var_u_scalar: float,
    var_v_scalar: float,
    var_y_scalar: float,
    std_U_safe: torch.Tensor,
    std_V_safe: torch.Tensor,
):
    """
    Convert physical variances into standardized diffusion coordinates.
    """
    VAR_U_used = standardize_variance(var_u_scalar, std_U_safe)
    VAR_V_used = standardize_variance(var_v_scalar, std_V_safe)
    VAR_Y_used = standardize_variance(var_y_scalar, std_V_safe)

    return (
        VAR_U_used.to(torch.float32),
        VAR_V_used.to(torch.float32),
        VAR_Y_used.to(torch.float32),
    )

# ============================================================
# (A) Load training dataset + preprocessing  [TIMED]
# ============================================================
t0 = time.time()

x_path = os.path.join(data_dir, "b_mn_samples_20000.npy")
y_path = os.path.join(data_dir, "sampled_solutions_20000_10_locs.npy")

x_np = np.load(x_path)[:K, :]
y_np = np.load(y_path)[:K, :]

x_sample = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).reshape(K, -1)
y_sample = torch.tensor(y_np, dtype=torch.float32, device=DEVICE).reshape(K, dim_v)

if ADD_NOISE_TO_TRAIN_V:
    noise = torch.randn(K, dim_v, device=DEVICE) * float(y_std)
    y_sample = y_sample + noise

# Save raw training data (physical)
np.save(os.path.join(savedir, "sample_U.npy"), x_sample.detach().cpu().numpy())
np.save(os.path.join(savedir, "sample_V.npy"), y_sample.detach().cpu().numpy())

# Normalization stats
sample_U = x_sample
sample_V = y_sample

mean_U = torch.mean(sample_U, dim=0)
std_U = torch.std(sample_U, dim=0)
mean_V = torch.mean(sample_V, dim=0)
std_V = torch.std(sample_V, dim=0)

eps_std = 1e-12
std_U_safe = torch.clamp(std_U, min=eps_std)
std_V_safe = torch.clamp(std_V, min=eps_std)

sample_U_normalized = (sample_U - mean_U) / std_U_safe
sample_V_normalized = (sample_V - mean_V) / std_V_safe

# Print physical-space scale diagnostics
print_block_scale_stats("U", sample_U)
print_block_scale_stats("V", sample_V)

# Build diffusion variances according to knob
VAR_U_gen, VAR_V_gen, VAR_Y_gen = build_diffusion_variances(
    var_u_scalar=VAR_U,
    var_v_scalar=VAR_V,
    var_y_scalar=VAR_Y,
    std_U_safe=std_U_safe,
    std_V_safe=std_V_safe,
)

# Select random conditions from training V
cond_idx = rng.choice(K, size=N_COND_TRAIN, replace=False)
cond_idx_t = torch.tensor(cond_idx, device=DEVICE, dtype=torch.long)

cond_Y_train_phys = sample_V[cond_idx_t]
cond_Y_train_norm = (cond_Y_train_phys - mean_V) / std_V_safe

np.save(
    os.path.join(savedir, f"cond_Y_train_phys_{N_COND_TRAIN}_{tag_hash}.npy"),
    cond_Y_train_phys.detach().cpu().numpy(),
)
np.save(
    os.path.join(savedir, f"cond_Y_train_norm_{N_COND_TRAIN}_{tag_hash}.npy"),
    cond_Y_train_norm.detach().cpu().numpy(),
)
np.save(
    os.path.join(savedir, f"cond_idx_train_{N_COND_TRAIN}_{tag_hash}.npy"),
    cond_idx,
)

t_load_prep = time.time() - t0
print(f"[timing] load+prep (incl noise+norm stats+cond select+saves) = {t_load_prep:.3f} s")

# ============================================================
# Variance report
# ============================================================
print("\n=== Diffusion variance mode ===")
print(f"VAR_MODE = {VAR_MODE}")

print("\nPhysical-space scalar knobs:")
print(f"VAR_U = {VAR_U}")
print(f"VAR_V = {VAR_V}")
print(f"VAR_Y = {VAR_Y}")

print("\nstd_U_safe:")
print(std_U_safe)
print("std_V_safe:")
print(std_V_safe)

print("\nPhysical-space variance relative to block scale:")
print(f"VAR_U / mean(std_U^2) = {VAR_U / (std_U_safe.pow(2).mean().item()):.6e}")
print(f"VAR_V / mean(std_V^2) = {VAR_V / (std_V_safe.pow(2).mean().item()):.6e}")
print(f"VAR_Y / mean(std_V^2) = {VAR_Y / (std_V_safe.pow(2).mean().item()):.6e}")

print("\nUsing PHYSICAL variances converted to normalized-space per dimension:")
print("VAR_U_gen = VAR_U / std_U_safe^2:")
print(VAR_U_gen)
print("VAR_V_gen = VAR_V / std_V_safe^2:")
print(VAR_V_gen)
print("VAR_Y_gen = VAR_Y / std_V_safe^2:")
print(VAR_Y_gen)

print("\nVariance summaries:")
print(
    f"VAR_U_gen: mean={VAR_U_gen.mean().item():.6e}, "
    f"min={VAR_U_gen.min().item():.6e}, max={VAR_U_gen.max().item():.6e}"
)
print(
    f"VAR_V_gen: mean={VAR_V_gen.mean().item():.6e}, "
    f"min={VAR_V_gen.min().item():.6e}, max={VAR_V_gen.max().item():.6e}"
)
print(
    f"VAR_Y_gen: mean={VAR_Y_gen.mean().item():.6e}, "
    f"min={VAR_Y_gen.min().item():.6e}, max={VAR_Y_gen.max().item():.6e}"
)

variance_report = {
    "device": DEVICE,
    "tag": tag,
    "tag_hash": tag_hash,
    "var_mode": VAR_MODE,
    "data_source": {
        "x_path": x_path,
        "y_path": y_path,
        "note": "Base dataset is loaded from disk; not generated from scratch in this script.",
    },
    "physical_variances": {
        "VAR_U": float(VAR_U),
        "VAR_V": float(VAR_V),
        "VAR_Y": float(VAR_Y),
    },
    "physical_scale_diagnostics": {
        "U_mean": mean_U.detach().cpu().numpy().tolist(),
        "U_std": std_U_safe.detach().cpu().numpy().tolist(),
        "V_mean": mean_V.detach().cpu().numpy().tolist(),
        "V_std": std_V_safe.detach().cpu().numpy().tolist(),
        "mean_stdU_sq": float(std_U_safe.pow(2).mean().item()),
        "mean_stdV_sq": float(std_V_safe.pow(2).mean().item()),
        "VAR_U_over_mean_stdU_sq": float(VAR_U / std_U_safe.pow(2).mean().item()),
        "VAR_V_over_mean_stdV_sq": float(VAR_V / std_V_safe.pow(2).mean().item()),
        "VAR_Y_over_mean_stdV_sq": float(VAR_Y / std_V_safe.pow(2).mean().item()),
    },
    "normalized_variances_used": {
        "VAR_U_gen": VAR_U_gen.detach().cpu().numpy().tolist(),
        "VAR_V_gen": VAR_V_gen.detach().cpu().numpy().tolist(),
        "VAR_Y_gen": VAR_Y_gen.detach().cpu().numpy().tolist(),
    },
}

with open(variance_json, "w") as f:
    json.dump(variance_report, f, indent=2)

with open(variance_txt, "w") as f:
    f.write("Variance report\n")
    f.write(f"tag: {tag}\n")
    f.write(f"tag_hash: {tag_hash}\n")
    f.write(f"VAR_MODE: {VAR_MODE}\n\n")

    f.write("Data source:\n")
    f.write(f"  x_path: {x_path}\n")
    f.write(f"  y_path: {y_path}\n")
    f.write("  note: Base dataset is loaded from disk; not generated from scratch in this script.\n\n")

    f.write("Physical-space scalar variance knobs:\n")
    f.write(f"  VAR_U = {VAR_U}\n")
    f.write(f"  VAR_V = {VAR_V}\n")
    f.write(f"  VAR_Y = {VAR_Y}\n\n")

    f.write("Scale diagnostics:\n")
    f.write(f"  mean(std_U^2) = {std_U_safe.pow(2).mean().item():.10e}\n")
    f.write(f"  mean(std_V^2) = {std_V_safe.pow(2).mean().item():.10e}\n")
    f.write(f"  VAR_U / mean(std_U^2) = {VAR_U / std_U_safe.pow(2).mean().item():.10e}\n")
    f.write(f"  VAR_V / mean(std_V^2) = {VAR_V / std_V_safe.pow(2).mean().item():.10e}\n")
    f.write(f"  VAR_Y / mean(std_V^2) = {VAR_Y / std_V_safe.pow(2).mean().item():.10e}\n\n")

    f.write("std_U_safe:\n")
    f.write(np.array2string(std_U_safe.detach().cpu().numpy(), precision=8))
    f.write("\n\n")

    f.write("std_V_safe:\n")
    f.write(np.array2string(std_V_safe.detach().cpu().numpy(), precision=8))
    f.write("\n\n")

    f.write("Normalized-space variances actually used:\n")
    f.write("VAR_U_gen:\n")
    f.write(np.array2string(VAR_U_gen.detach().cpu().numpy(), precision=8))
    f.write("\n\n")
    f.write("VAR_V_gen:\n")
    f.write(np.array2string(VAR_V_gen.detach().cpu().numpy(), precision=8))
    f.write("\n\n")
    f.write("VAR_Y_gen:\n")
    f.write(np.array2string(VAR_Y_gen.detach().cpu().numpy(), precision=8))
    f.write("\n")

print(f"[saved] variance json -> {variance_json}")
print(f"[saved] variance txt  -> {variance_txt}")

# ============================================================
# (B) DM generation for TRAINING LABELS  [TIMED]
# ============================================================
gen_sample_size = N_COND_TRAIN
num_batches = int(np.ceil(gen_sample_size / DM_BATCH))

xT = torch.randn(gen_sample_size, dim_x, device=DEVICE)
torch.save(xT, os.path.join(savedir, f"xT_dm_train_{gen_sample_size}_{tag_hash}.pt"))

samples_regen_list = []

cuda_sync()
t0 = time.time()

for batch_idx in range(num_batches):
    start = batch_idx * DM_BATCH
    end = min((batch_idx + 1) * DM_BATCH, gen_sample_size)

    x_T_batch = xT[start:end]
    cond_Y_batch = cond_Y_train_norm[start:end]

    score_normal_cond_batch = partial(
        conditional_score,
        sample_u=sample_U_normalized,
        sample_v=sample_V_normalized,
        condition_y=cond_Y_batch,
        variance_u=VAR_U_gen,
        variance_v=VAR_V_gen,
        variance_y=VAR_Y_gen,
    )

    samples_batch = reverse_sde(
        x_terminal=x_T_batch,
        time_steps=TIME_STEPS,
        drift=b,
        diffusion=sigma,
        score=score_normal_cond_batch,
        save_path=False,
    )

    samples_regen_list.append(samples_batch)
    print(f"[DM-train] Batch {batch_idx+1}/{num_batches}  [{start}:{end}]")

samples_regen_norm = torch.cat(samples_regen_list, dim=0)

samples_regen_phys = samples_regen_norm.clone()
samples_regen_phys[:, 0:dim_u] = (samples_regen_phys[:, 0:dim_u] * std_U_safe) + mean_U
samples_regen_phys[:, dim_u:dim_u + dim_v] = (samples_regen_phys[:, dim_u:dim_u + dim_v] * std_V_safe) + mean_V

dm_out_path = os.path.join(
    savedir,
    f"samples_regen_dmtrain_{gen_sample_size}_VARMODE_{VAR_MODE}_VAR_Y_{VAR_Y}_VAR_U_{VAR_U}_VAR_V_{VAR_V}_"
    f"yStd_{y_std if ADD_NOISE_TO_TRAIN_V else 0.0}_{tag_hash}_K_{K}.npy",
)
np.save(dm_out_path, samples_regen_phys.detach().cpu().numpy())

cuda_sync()
t_dm_total = time.time() - t0
t_dm_per = t_dm_total / float(gen_sample_size)
print(f"[timing] DM label generation total = {t_dm_total:.3f} s  ({t_dm_per:.6e} s/sample)")
print("[DM-train] Saved:", dm_out_path)

# ============================================================
# (C) Build FN training tensors + normalize  [TIMED]
# ============================================================
cuda_sync()
t0 = time.time()

yTrain = torch.hstack(
    [cond_Y_train_phys.reshape(-1, dim_v), xT.reshape(-1, dim_x)]
).to(DEVICE)
xTrain = samples_regen_phys[:, 0:dim_u].reshape(-1, dim_u).to(DEVICE)

xTrain_mean = xTrain.mean(dim=0, keepdim=True)
xTrain_std = xTrain.std(dim=0, keepdim=True).clamp(min=1e-12)
yTrain_mean = yTrain.mean(dim=0, keepdim=True)
yTrain_std = yTrain.std(dim=0, keepdim=True).clamp(min=1e-12)

xTrain_norm = (xTrain - xTrain_mean) / xTrain_std
yTrain_norm = (yTrain - yTrain_mean) / yTrain_std

cuda_sync()
t_build = time.time() - t0
print(f"[timing] build+normalize FN tensors = {t_build:.3f} s  (N={yTrain_norm.shape[0]})")

# ============================================================
# (D) Train FN on ALL data  [TIMED]
# ============================================================
FN = FN_Net(dim_v + dim_x, dim_u, hid_size=n_neurons, n_layers=n_layers).to(DEVICE)
optimizer = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

training_loss = []
best_loss = float("inf")
best_state_dict = None
best_epoch = -1

model_save_path = os.path.join(savedir, f"FN_trained_model_{tag}_{tag_hash}_K_{K}.pth")
stats_save_path = os.path.join(savedir, f"FN_trained_model_{tag}_{tag_hash}_stats_K_{K}.pth")

print("\n[FN] Training...")
cuda_sync()
t0 = time.time()

for ep in range(total_epochs):
    FN.train()
    optimizer.zero_grad()
    pred = FN(yTrain_norm)
    loss = criterion(pred, xTrain_norm)
    loss.backward()
    optimizer.step()

    loss_val = float(loss.item())
    training_loss.append(loss_val)

    if loss_val < best_loss:
        best_loss = loss_val
        best_epoch = ep
        best_state_dict = {
            "model_state_dict": FN.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": ep,
            "best_loss": best_loss,
            "tag": tag,
            "tag_hash": tag_hash,
        }

    if ep % 100 == 0:
        print(f"[FN] Epoch {ep:6d} | loss={loss_val:.6e} | best={best_loss:.6e} @ {best_epoch}")

cuda_sync()
t_train_total = time.time() - t0
t_train_per_epoch = t_train_total / float(total_epochs)
print(f"[timing] FN training total = {t_train_total:.3f} s  ({t_train_per_epoch:.6e} s/epoch)")

torch.save(best_state_dict, model_save_path)
torch.save(
    {
        "xTrain_mean": xTrain_mean,
        "xTrain_std": xTrain_std,
        "yTrain_mean": yTrain_mean,
        "yTrain_std": yTrain_std,
        "tag": tag,
        "tag_hash": tag_hash,
        "VAR_MODE": VAR_MODE,
        "VAR_Y": VAR_Y,
        "VAR_U": VAR_U,
        "VAR_V": VAR_V,
        "VAR_U_gen": VAR_U_gen.detach().cpu(),
        "VAR_V_gen": VAR_V_gen.detach().cpu(),
        "VAR_Y_gen": VAR_Y_gen.detach().cpu(),
        "mean_U": mean_U.detach().cpu(),
        "std_U_safe": std_U_safe.detach().cpu(),
        "mean_V": mean_V.detach().cpu(),
        "std_V_safe": std_V_safe.detach().cpu(),
        "y_std": float(y_std if ADD_NOISE_TO_TRAIN_V else 0.0),
        "ADD_NOISE_TO_TRAIN_V": bool(ADD_NOISE_TO_TRAIN_V),
        "TIME_STEPS": TIME_STEPS,
        "DM_BATCH": DM_BATCH,
    },
    stats_save_path,
)

print("\n[FN] Saved best model:", model_save_path)
print("[FN] Saved stats:", stats_save_path)

# Loss plot
t0 = time.time()
training_loss_np = np.array(training_loss, dtype=np.float64)
plt.figure()
plt.plot(np.arange(total_epochs), training_loss_np)
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Elliptic PDE FN training loss")
plt.tight_layout()
loss_png = os.path.join(savedir, f"FN_training_loss_{tag}_{tag_hash}_K_{K}.png")
plt.savefig(loss_png, dpi=200)
plt.close()
t_lossplot = time.time() - t0
print("[FN] Saved loss plot:", loss_png)
print(f"[timing] loss plot save = {t_lossplot:.3f} s")

# ============================================================
# (E) Reload FN + stats
# ============================================================
FN = FN_Net(dim_v + dim_x, dim_u, hid_size=n_neurons, n_layers=n_layers).to(DEVICE)
checkpoint = torch.load(model_save_path, map_location=DEVICE)
FN.load_state_dict(checkpoint["model_state_dict"])
FN.eval()

stats = torch.load(stats_save_path, map_location=DEVICE)
xTrain_mean = stats["xTrain_mean"].to(DEVICE)
xTrain_std = stats["xTrain_std"].to(DEVICE)
yTrain_mean = stats["yTrain_mean"].to(DEVICE)
yTrain_std = stats["yTrain_std"].to(DEVICE)
mean_U = stats["mean_U"].to(DEVICE)
std_U_safe = stats["std_U_safe"].to(DEVICE)
mean_V = stats["mean_V"].to(DEVICE)
std_V_safe = stats["std_V_safe"].to(DEVICE)
VAR_U_gen = stats["VAR_U_gen"].to(DEVICE)
VAR_V_gen = stats["VAR_V_gen"].to(DEVICE)
VAR_Y_gen = stats["VAR_Y_gen"].to(DEVICE)

# ============================================================
# NN inference timing helpers
# ============================================================
def build_y_in_for_testcase(testcase_id: int, n_samples: int, rng_local: np.random.Generator):
    testcase_path = os.path.join(data_dir, f"testcase_{testcase_id}.npy")
    if not os.path.exists(testcase_path):
        raise FileNotFoundError(f"Missing testcase file: {testcase_path}")

    y_obs = np.load(testcase_path).reshape(1, dim_v)
    y_obs_rep = np.repeat(y_obs, repeats=n_samples, axis=0).astype(np.float32)
    zT = rng_local.standard_normal(size=(n_samples, dim_x)).astype(np.float32)
    y_in = np.hstack([y_obs_rep, zT]).astype(np.float32)
    return torch.tensor(y_in, dtype=torch.float32, device=DEVICE)

def pure_infer_benchmark(FN_model: nn.Module, y_in_t: torch.Tensor):
    FN_model.eval()
    with torch.no_grad():
        for _ in range(INFER_WARMUP):
            out_norm = FN_model((y_in_t - yTrain_mean) / yTrain_std)
            _ = out_norm * xTrain_std + xTrain_mean
    cuda_sync()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(INFER_REPEATS):
            out_norm = FN_model((y_in_t - yTrain_mean) / yTrain_std)
            _ = out_norm * xTrain_std + xTrain_mean
    cuda_sync()
    return (time.time() - t0) / float(INFER_REPEATS)

def nn_sample_for_testcase(testcase_id: int, n_samples: int, rng_local: np.random.Generator):
    y_in_t = build_y_in_for_testcase(testcase_id, n_samples, rng_local)
    with torch.no_grad():
        out_norm = FN((y_in_t - yTrain_mean) / yTrain_std)
        out_phys = out_norm * xTrain_std + xTrain_mean
    return out_phys.detach().cpu().numpy().reshape(n_samples, dim_u)

# ============================================================
# (F) NN inference timing + save NN samples
# ============================================================
rng_inf = np.random.default_rng(SEED + 12345)

y_in_t_bench = build_y_in_for_testcase(TESTCASES_TO_TEST[0], N_NN_SAMPLES, rng_inf)
t_infer_avg_batch = pure_infer_benchmark(FN, y_in_t_bench)
t_infer_avg_sample = t_infer_avg_batch / float(N_NN_SAMPLES)
print(f"[timing] PURE NN inference avg = {t_infer_avg_batch:.6e} s/batch  ({t_infer_avg_sample:.6e} s/sample)  batch={N_NN_SAMPLES}")

print("\n[NN] Generating samples for testcases:", TESTCASES_TO_TEST)
for tc in TESTCASES_TO_TEST:
    t0 = time.time()
    samples_tc = nn_sample_for_testcase(tc, N_NN_SAMPLES, rng_inf)
    nn_out_path = os.path.join(
        savedir,
        f"NN_output_testcase_{tc}_VARMODE_{VAR_MODE}_VAR_Y_{VAR_Y}_VAR_U_{VAR_U}_VAR_V_{VAR_V}_"
        f"Ntrain_{N_COND_TRAIN}_yStd_{y_std if ADD_NOISE_TO_TRAIN_V else 0.0}_"
        f"FNw{n_neurons}_FNlayers{n_layers}_K_{K}.npy",
    )
    np.save(nn_out_path, samples_tc)
    t_tc = time.time() - t0
    t_infer_e2e[tc] = t_tc
    print(f"[NN] Saved testcase_{tc} -> {nn_out_path} | shape={samples_tc.shape} | e2e_time={t_tc:.3f}s")

# ============================================================
# (G) PURE DIFFUSION inference conditioned on testcase(s)
# ============================================================
def dm_generate_conditioned_on_testcase(test_case: int,
                                        n_samples: int,
                                        time_steps: int,
                                        batch_size: int):
    testcase_path = os.path.join(data_dir, f"testcase_{test_case}.npy")
    if not os.path.exists(testcase_path):
        raise FileNotFoundError(f"Missing testcase file: {testcase_path}")

    gen_sample_size = int(n_samples)
    num_batches = int(np.ceil(gen_sample_size / int(batch_size)))

    cuda_sync()
    t0_all = time.time()

    y_obs_np = np.load(testcase_path).reshape(1, dim_v)
    y_obs = torch.tensor(y_obs_np, dtype=torch.float32, device=DEVICE)

    cond_Y_phys = y_obs.repeat(gen_sample_size, 1)
    cond_Y_norm = (cond_Y_phys - mean_V) / std_V_safe

    xT_test = torch.randn(gen_sample_size, dim_u + dim_v, device=DEVICE)

    VAR_U_local = VAR_U_gen
    VAR_V_local = VAR_V_gen
    VAR_Y_local = VAR_Y_gen

    samples_list = []
    for batch_idx in range(num_batches):
        start = batch_idx * int(batch_size)
        end = min((batch_idx + 1) * int(batch_size), gen_sample_size)

        x_T_batch = xT_test[start:end]
        cond_Y_batch = cond_Y_norm[start:end]

        score_batch = partial(
            conditional_score,
            sample_u=sample_U_normalized,
            sample_v=sample_V_normalized,
            condition_y=cond_Y_batch,
            variance_u=VAR_U_local,
            variance_v=VAR_V_local,
            variance_y=VAR_Y_local,
        )

        samples_batch = reverse_sde(
            x_terminal=x_T_batch,
            time_steps=int(time_steps),
            drift=b,
            diffusion=sigma,
            score=score_batch,
            save_path=False,
        )

        samples_list.append(samples_batch)
        print(f"[DM-testcase{test_case}] Batch {batch_idx+1}/{num_batches}  [{start}:{end}]")

    samples_norm = torch.cat(samples_list, dim=0)

    samples_phys = samples_norm.clone()
    samples_phys[:, 0:dim_u] = (samples_phys[:, 0:dim_u] * std_U_safe) + mean_U
    samples_phys[:, dim_u:dim_u + dim_v] = (samples_phys[:, dim_u:dim_u + dim_v] * std_V_safe) + mean_V

    out_path = os.path.join(
        savedir,
        f"DM_pure_samples_testcase_{test_case}_Nsamp_{gen_sample_size}_"
        f"VARMODE_{VAR_MODE}_VAR_Y_{VAR_Y}_VAR_U_{VAR_U}_VAR_V_{VAR_V}_"
        f"yStdTrain_{y_std if ADD_NOISE_TO_TRAIN_V else 0.0}_"
        f"T_{time_steps}_B_{batch_size}_K_{K}.npy",
    )
    np.save(out_path, samples_phys.detach().cpu().numpy())

    cuda_sync()
    t_total = time.time() - t0_all
    t_per_sample = t_total / float(gen_sample_size)
    return out_path, t_total, t_per_sample

print("\n[DM-pure] Generating PURE diffusion samples for testcases:", TESTCASES_TO_TEST)
for tc in TESTCASES_TO_TEST:
    out_path, t_total, t_per_sample = dm_generate_conditioned_on_testcase(
        test_case=tc,
        n_samples=N_DM_SAMPLES_TEST,
        time_steps=DM_TEST_TIME_STEPS,
        batch_size=DM_TEST_BATCH,
    )
    dm_test_out_path[tc] = out_path
    t_dm_test_total[tc] = t_total
    t_dm_test_per[tc] = t_per_sample
    print(f"[timing] PURE DM testcase_{tc}: total={t_total:.3f}s  ({t_per_sample:.6e}s/sample) | saved={out_path}")

# ============================================================
# (H) Write timing CSV
# ============================================================
with open(timing_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "tag", "tag_hash", "device",
        "K", "N_COND_TRAIN", "TIME_STEPS_trainDM", "DM_BATCH_trainDM",
        "VAR_MODE",
        "VAR_Y_physical", "VAR_U_physical", "VAR_V_physical", "y_std_trainV", "ADD_NOISE_TO_TRAIN_V",
        "VAR_U_gen_mean", "VAR_U_gen_min", "VAR_U_gen_max",
        "VAR_V_gen_mean", "VAR_V_gen_min", "VAR_V_gen_max",
        "VAR_Y_gen_mean", "VAR_Y_gen_min", "VAR_Y_gen_max",
        "FN_width", "FN_layers", "epochs", "lr",
        "t_load_prep_s",
        "t_dm_label_total_s", "t_dm_label_per_sample_s",
        "t_build_train_tensors_s",
        "t_train_total_s", "t_train_per_epoch_s",
        "best_epoch", "best_train_loss",
        "t_lossplot_save_s",
        "t_pure_nn_infer_avg_s_per_batch", "t_pure_nn_infer_avg_s_per_sample",
        "nn_infer_batch_size",
        "t_nn_e2e_testcase_1_s", "t_nn_e2e_testcase_2_s",
        "DM_pure_Nsamp_test", "DM_pure_time_steps", "DM_pure_batch",
        "t_dm_pure_testcase_1_total_s", "t_dm_pure_testcase_1_per_sample_s",
        "t_dm_pure_testcase_2_total_s", "t_dm_pure_testcase_2_per_sample_s",
        "dm_train_out_path", "fn_model_path", "fn_stats_path", "fn_loss_png",
        "dm_pure_out_path_testcase_1", "dm_pure_out_path_testcase_2",
        "variance_json", "variance_txt",
    ])
    w.writerow([
        tag, tag_hash, DEVICE,
        K, N_COND_TRAIN, TIME_STEPS, DM_BATCH,
        VAR_MODE,
        VAR_Y, VAR_U, VAR_V, float(y_std), bool(ADD_NOISE_TO_TRAIN_V),
        f"{VAR_U_gen.mean().item():.9e}", f"{VAR_U_gen.min().item():.9e}", f"{VAR_U_gen.max().item():.9e}",
        f"{VAR_V_gen.mean().item():.9e}", f"{VAR_V_gen.min().item():.9e}", f"{VAR_V_gen.max().item():.9e}",
        f"{VAR_Y_gen.mean().item():.9e}", f"{VAR_Y_gen.min().item():.9e}", f"{VAR_Y_gen.max().item():.9e}",
        n_neurons, n_layers, total_epochs, LEARNING_RATE,
        f"{t_load_prep:.6f}",
        f"{t_dm_total:.6f}", f"{t_dm_per:.9e}",
        f"{t_build:.6f}",
        f"{t_train_total:.6f}", f"{t_train_per_epoch:.9e}",
        best_epoch, f"{best_loss:.9e}",
        f"{t_lossplot:.6f}",
        f"{t_infer_avg_batch:.9e}", f"{t_infer_avg_sample:.9e}",
        N_NN_SAMPLES,
        f"{t_infer_e2e.get(1, float('nan')):.6f}",
        f"{t_infer_e2e.get(2, float('nan')):.6f}",
        N_DM_SAMPLES_TEST, DM_TEST_TIME_STEPS, DM_TEST_BATCH,
        f"{t_dm_test_total.get(1, float('nan')):.6f}",
        f"{t_dm_test_per.get(1, float('nan')):.9e}",
        f"{t_dm_test_total.get(2, float('nan')):.6f}",
        f"{t_dm_test_per.get(2, float('nan')):.9e}",
        dm_out_path, model_save_path, stats_save_path, loss_png,
        dm_test_out_path.get(1, ""),
        dm_test_out_path.get(2, ""),
        variance_json, variance_txt,
    ])

print("\nAll done.")
print("[timing] CSV saved:", timing_csv)
print("[variance] JSON saved:", variance_json)
print("[variance] TXT saved:", variance_txt)
