#!/usr/bin/env python3
"""Run surrogate-based MCMC for the elliptic PDE example.

The PyTorch surrogate is trained on the generated dataset before emcee is run
for both test cases. Autocorrelation-thinned samples and convergence
diagnostics are saved under the selected output directory.
"""

import argparse
import os
import time
import json
import numpy as np
import emcee


parser = argparse.ArgumentParser(description="Run elliptic PDE surrogate MCMC.")
parser.add_argument(
    "--output-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "elliptic_pde_results"),
    help="Directory containing data/ and receiving MCMC results.",
)
args = parser.parse_args()

# ----------------------------
# PyTorch surrogate
# ----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = repr(e)

# ----------------------------
# USER SETTINGS
# ----------------------------
savedir = os.path.abspath(args.output_dir)
data_dir = os.path.join(savedir, "data")
out_root = os.path.join(savedir, "elliptic_pde_mcmc")
os.makedirs(out_root, exist_ok=True)

SEED = 42
np.random.seed(SEED)

dim_u = 4
dim_v = 10
TESTCASE_IDS = [1,2]

sigma_y = 0.0025
var_y = sigma_y**2

nwalkers = 32
burnin_steps = 10000
prod_steps = 25000

N_POST = 5000                 # maximum number of thinned samples to save
CONV_WINDOW = 50              # for early/late diagnostics

# Diagnostic controls
ROLL_BLOCKS = 5

B_ABS_MAX = 6.0

# ----------------------------
# SURROGATE TRAINING SETTINGS
# ----------------------------
SURROGATE_NTRAIN = 20000
SURR_BATCH = 256
SURR_MAX_EPOCHS = 5000
SURR_LR = 1e-3
SURR_WEIGHT_DECAY = 1e-6
SURR_VAL_FRAC = 0.15
SURR_PATIENCE = 50
SURR_HID = 128
SURR_DEPTH = 3
SURR_ACT = "tanh"  # "relu" or "tanh"

USE_GPU_SURROGATE = True

B_PATH = os.path.join(data_dir, "b_mn_samples_20000.npy")
Y_PATH = os.path.join(data_dir, "sampled_solutions_20000_10_locs.npy")

# Prior
prior_mu = np.zeros(dim_u)
prior_cov = np.diag([1/2, 1/3, 1/3, 1/4])
prior_cov_inv = np.diag(1/np.diag(prior_cov))
prior_logdet = np.sum(np.log(np.diag(prior_cov)))
prior_normconst = -0.5 * (dim_u * np.log(2*np.pi) + prior_logdet)

# ----------------------------
# Helpers
# ----------------------------
def last_k(arr, k):
    arr = np.asarray(arr)
    if arr.shape[0] <= k:
        return arr
    return arr[-k:]


def mean_min_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.min(x)), float(np.max(x))


def safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def safe_list_float(arr):
    if arr is None:
        return None
    out = []
    for x in arr:
        out.append(safe_float(x))
    return out


def split_rhat_from_walkers(chain):
    """
    Split-Rhat using walkers as chains.
    chain: (steps, walkers, dim)
    Split each walker into two halves => 2*walkers chains of length steps//2
    """
    chain = np.asarray(chain)
    nsteps, nwalk, ndim = chain.shape
    n = nsteps // 2
    if n < 20:
        return None

    first = chain[:n, :, :]
    second = chain[-n:, :, :]
    x = np.concatenate(
        [np.transpose(first, (1, 0, 2)), np.transpose(second, (1, 0, 2))],
        axis=0
    )  # (2w, n, d)
    m = x.shape[0]

    chain_means = np.mean(x, axis=1)
    chain_vars = np.var(x, axis=1, ddof=1)

    W = np.mean(chain_vars, axis=0)
    mean_of_means = np.mean(chain_means, axis=0)
    B = (n * np.sum((chain_means - mean_of_means) ** 2, axis=0)) / (m - 1)

    var_hat = ((n - 1) / n) * W + (1 / n) * B
    rhat = np.sqrt(np.where(W > 0, var_hat / W, np.nan))
    rhat = np.where(np.isfinite(rhat), rhat, np.nan)
    return rhat


def chunk_indices(nsteps, nblocks):
    nblocks = max(1, int(nblocks))
    edges = np.linspace(0, nsteps, nblocks + 1, dtype=int)
    return [(edges[i], edges[i + 1]) for i in range(nblocks) if edges[i + 1] > edges[i]]


# ----------------------------
# Surrogate definition/training
# ----------------------------
class MLP(nn.Module):
    def __init__(self, din, dout, hid=256, depth=4, act="tanh"):
        super().__init__()
        if act.lower() == "relu":
            Act = nn.ReLU
        else:
            Act = nn.Tanh

        layers = []
        if depth <= 1:
            layers = [nn.Linear(din, dout)]
        else:
            layers.append(nn.Linear(din, hid))
            layers.append(Act())
            for _ in range(depth - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(Act())
            layers.append(nn.Linear(hid, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_surrogate_and_get_predictor(out_root_dir):
    if not TORCH_OK:
        raise RuntimeError(f"PyTorch import failed: {TORCH_ERR}")

    device = "cuda" if (USE_GPU_SURROGATE and torch.cuda.is_available()) else "cpu"

    # Load training data (first 10k)
    B = np.load(B_PATH)[:SURROGATE_NTRAIN, :].astype(np.float32)
    Y = np.load(Y_PATH)[:SURROGATE_NTRAIN, :].astype(np.float32)

    assert B.shape[1] == dim_u, f"Expected B dim {dim_u}, got {B.shape}"
    assert Y.shape[1] == dim_v, f"Expected Y dim {dim_v}, got {Y.shape}"

    # Split train/val
    n = B.shape[0]
    n_val = int(round(SURR_VAL_FRAC * n))
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    B_tr, Y_tr = B[tr_idx], Y[tr_idx]
    B_va, Y_va = B[val_idx], Y[val_idx]

    # Standardize
    b_mean = B_tr.mean(axis=0)
    b_std = B_tr.std(axis=0)
    b_std = np.where(b_std > 0, b_std, 1.0)

    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0)
    y_std = np.where(y_std > 0, y_std, 1.0)

    Xtr = (B_tr - b_mean) / b_std
    Ztr = (Y_tr - y_mean) / y_std
    Xva = (B_va - b_mean) / b_std
    Zva = (Y_va - y_mean) / y_std

    Xtr_t = torch.from_numpy(Xtr)
    Ztr_t = torch.from_numpy(Ztr)
    Xva_t = torch.from_numpy(Xva)
    Zva_t = torch.from_numpy(Zva)

    set_torch_seed(SEED)
    model = MLP(dim_u, dim_v, hid=SURR_HID, depth=SURR_DEPTH, act=SURR_ACT).to(device)

    opt = optim.Adam(model.parameters(), lr=SURR_LR, weight_decay=SURR_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    n_tr = Xtr_t.shape[0]
    best_val = np.inf
    best_state = None
    patience = 0

    t_train0 = time.time()
    train_log = []

    for epoch in range(1, SURR_MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr)
        Xtr_ep = Xtr_t[perm]
        Ztr_ep = Ztr_t[perm]

        tr_loss_acc = 0.0
        nb = 0
        for i in range(0, n_tr, SURR_BATCH):
            xb = Xtr_ep[i:i+SURR_BATCH].to(device)
            zb = Ztr_ep[i:i+SURR_BATCH].to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, zb)
            loss.backward()
            opt.step()

            tr_loss_acc += float(loss.detach().cpu().item())
            nb += 1

        tr_loss = tr_loss_acc / max(1, nb)

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t.to(device))
            val_loss = float(loss_fn(pred_va, Zva_t.to(device)).detach().cpu().item())

        train_log.append({"epoch": int(epoch), "train_mse": tr_loss, "val_mse": val_loss})

        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % 50 == 0:
            print(f"[SURROGATE] epoch {epoch:4d} | train_mse={tr_loss:.6e} | val_mse={val_loss:.6e} | best_val={best_val:.6e} | device={device}")

        if patience >= SURR_PATIENCE:
            print(f"[SURROGATE] early stopping at epoch {epoch} (patience {SURR_PATIENCE})")
            break

    t_train = time.time() - t_train0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save surrogate artifacts
    surr_dir = os.path.join(out_root_dir, "surrogate")
    os.makedirs(surr_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {"din": dim_u, "dout": dim_v, "hid": SURR_HID, "depth": SURR_DEPTH, "act": SURR_ACT},
            "train_settings": {
                "ntrain": SURROGATE_NTRAIN, "batch": SURR_BATCH, "max_epochs": SURR_MAX_EPOCHS,
                "lr": SURR_LR, "weight_decay": SURR_WEIGHT_DECAY, "val_frac": SURR_VAL_FRAC,
                "patience": SURR_PATIENCE, "seed": SEED, "device": device,
            },
            "best_val_mse": float(best_val),
            "train_time_s": float(t_train),
        },
        os.path.join(surr_dir, "surrogate_state.pt")
    )

    with open(os.path.join(surr_dir, "scaler_stats.json"), "w") as f:
        json.dump({"b_mean": b_mean.tolist(), "b_std": b_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist()}, f, indent=2)

    with open(os.path.join(surr_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    b_mean_t = torch.from_numpy(b_mean).to(device)
    b_std_t = torch.from_numpy(b_std).to(device)
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)

    model.eval()

    def predict_y_numpy(b_np_1d):
        xb = torch.as_tensor(b_np_1d, dtype=torch.float32, device=device).view(1, -1)
        xb = (xb - b_mean_t) / b_std_t
        with torch.no_grad():
            z = model(xb)
            yhat = z * y_std_t + y_mean_t
        return yhat.detach().cpu().numpy().reshape(-1)

    surrogate_info = {"device": device, "best_val_mse": float(best_val), "train_time_s": float(t_train), "surrogate_dir": surr_dir}
    return predict_y_numpy, surrogate_info


# ----------------------------
# Train surrogate ONCE
# ----------------------------
print("====================================================")
print(f"Training elliptic PDE surrogate from first {SURROGATE_NTRAIN} samples...")
print("====================================================")
t0 = time.time()
predict_y, surrogate_info = train_surrogate_and_get_predictor(out_root)
t_surr_total = time.time() - t0
print(f"[SURROGATE] done. device={surrogate_info['device']} best_val_mse={surrogate_info['best_val_mse']:.6e} "
      f"train_time_s={surrogate_info['train_time_s']:.3f} total_setup_s={t_surr_total:.3f}")
print("")


# ----------------------------
# Main loop
# ----------------------------
for tc in TESTCASE_IDS:

    print("\n====================================================")
    print(f"Running testcase {tc}")
    print("====================================================")

    t_case0 = time.time()

    y_obs = np.load(os.path.join(data_dir, f"testcase_{tc}.npy")).reshape(dim_v).astype(float)

    def log_likelihood(b):
        y = predict_y(b)
        r = y_obs - y
        return -0.5 * (r @ r) / var_y - 0.5 * dim_v * np.log(2*np.pi*var_y)

    def log_post(b):
        if np.any(np.abs(b) > B_ABS_MAX):
            return -np.inf
        d = b - prior_mu
        lp = prior_normconst - 0.5 * (d @ prior_cov_inv @ d)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(b)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # ----------------------------
    # Burn-in
    # ----------------------------
    x0 = np.random.multivariate_normal(prior_mu, prior_cov, nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, dim_u, log_post)

    print("Burn-in...")
    t0 = time.time()
    state = sampler.run_mcmc(x0, burnin_steps, progress=True)
    t_burn = time.time() - t0

    burn_acc = sampler.acceptance_fraction.copy()
    burn_acc_mean, burn_acc_min, burn_acc_max = mean_min_max(burn_acc)

    sampler.reset()

    # ----------------------------
    # Production
    # ----------------------------
    print("Production...")
    t0 = time.time()
    sampler.run_mcmc(state, prod_steps, progress=True)
    t_prod = time.time() - t0

    prod_acc = sampler.acceptance_fraction.copy()
    prod_acc_mean, prod_acc_min, prod_acc_max = mean_min_max(prod_acc)

    # ----------------------------
    # Postproc + diagnostics + saving
    # ----------------------------
    out_dir = os.path.join(out_root, f"testcase_{tc}_sigma{sigma_y}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    chain = sampler.get_chain()        # (steps, walkers, dim)
    logp = sampler.get_log_prob()      # (steps, walkers)

    nsteps_eff, nwalkers_eff, _ = chain.shape
    N_total = int(nsteps_eff * nwalkers_eff)

    # Quick early/late diagnostics in STEPS domain
    W = int(min(CONV_WINDOW, nsteps_eff))
    logp_early = safe_float(np.mean(logp[:W, :]))
    logp_late = safe_float(np.mean(logp[-W:, :]))
    logp_delta = safe_float((logp_late - logp_early) if (logp_early is not None and logp_late is not None) else None)

    b_early = np.mean(chain[:W, :, :], axis=(0, 1))
    b_late = np.mean(chain[-W:, :, :], axis=(0, 1))
    b_delta = b_late - b_early

    b_early_list = safe_list_float(b_early)
    b_late_list = safe_list_float(b_late)
    b_delta_list = safe_list_float(b_delta)

    # Rolling blocks across production
    block_stats = []
    for (a, b) in chunk_indices(nsteps_eff, ROLL_BLOCKS):
        b_mean = np.mean(chain[a:b, :, :], axis=(0, 1))
        lp_mean = np.mean(logp[a:b, :])
        block_stats.append({"step_range": [int(a), int(b)], "mean_logp": safe_float(lp_mean), "mean_b": safe_list_float(b_mean)})

    # Split-Rhat
    rhat_list = None
    rhat_max = None
    try:
        rhat = split_rhat_from_walkers(chain)
        if rhat is not None:
            rhat_list = [None if np.isnan(x) else float(x) for x in rhat]
            finite = rhat[np.isfinite(rhat)]
            rhat_max = float(np.max(finite)) if finite.size > 0 else None
    except Exception:
        rhat_list = None
        rhat_max = None

    # Autocorrelation thinning and the only saved coefficient sample set.
    tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
    if not np.all(np.isfinite(tau) & (tau > 0.0)):
        raise RuntimeError(f"Invalid MCMC autocorrelation times: {tau}")

    ess = N_total / tau
    tau_max = float(np.max(tau))
    thin_ess = max(1, int(np.ceil(tau_max)))
    thinned = chain[::thin_ess, :, :].reshape(-1, dim_u)
    thinned_samples = last_k(thinned, N_POST)

    tau_list = [float(value) for value in tau]
    ess_list = [float(value) for value in ess]
    ess_min = float(np.min(ess))
    ess_mean = float(np.mean(ess))
    thinned_count = int(thinned_samples.shape[0])

    t_postproc = time.time() - t0

    t0 = time.time()
    sample_path = os.path.join(savedir, f"MCMC_samples_testcase{tc}.npy")
    np.save(sample_path, thinned_samples)
    t_save = time.time() - t0

    t_case = time.time() - t_case0

    # ----------------------------
    # PRINTS
    # ----------------------------
    print("\n[DIAGNOSTICS]")
    print(f"  production draws total (unthinned) : {N_total} = steps({nsteps_eff}) * walkers({nwalkers_eff})")
    print(f"  saved thinned samples              : {thinned_count}")
    print(f"  output                             : {sample_path}")
    print(f"  conv window W                      : {W} steps (per walker)")
    print(f"  mean(logp) early                   : {logp_early}")
    print(f"  mean(logp) late                    : {logp_late}")
    print(f"  delta(logp) late-early             : {logp_delta}")
    print(f"  mean(b) early                      : {b_early_list}")
    print(f"  mean(b) late                       : {b_late_list}")
    print(f"  delta(mean(b)) late-early          : {b_delta_list}")

    print("\n[ROLLING BLOCK MEANS] (to spot drift without plots)")
    for i, bs in enumerate(block_stats):
        a, b = bs["step_range"]
        print(f"  block {i+1}/{len(block_stats)} steps[{a}:{b}]  mean(logp)={bs['mean_logp']}  mean(b)={bs['mean_b']}")

    if rhat_list is None:
        print("\n[RHAT]")
        print("  split-Rhat                          : FAILED/UNAVAILABLE")
    else:
        print("\n[RHAT]")
        print(f"  split-Rhat per dim                  : {rhat_list}  (max={rhat_max})")

    print("\n[ESS]")
    print(f"  tau (IACT) per dim                 : {tau_list}  (tau_max={tau_max})")
    print(f"  ESS per dim                        : {ess_list}  (ESS min/mean={ess_min} / {ess_mean})")
    print(f"  thinning interval                  : {thin_ess}")

    # ----------------------------
    # TIMING
    # ----------------------------
    print("\n[TIMING]")
    print(f"  burn-in    : {t_burn:.3f} s   (steps={burnin_steps}, walkers={nwalkers})")
    print(f"  production : {t_prod:.3f} s   (steps={prod_steps}, walkers={nwalkers})")
    print(f"  postproc   : {t_postproc:.3f} s")
    print(f"  save       : {t_save:.3f} s")
    print(f"  TOTAL      : {t_case:.3f} s")
    print(f"  accept burn (mean/min/max): {burn_acc_mean:.3f} / {burn_acc_min:.3f} / {burn_acc_max:.3f}")
    print(f"  accept prod (mean/min/max): {prod_acc_mean:.3f} / {prod_acc_min:.3f} / {prod_acc_max:.3f}")

    # ----------------------------
    # SAVE JSON SUMMARY
    # ----------------------------
    summary = {
        "testcase": int(tc),
        "sigma_y": float(sigma_y),
        "var_y": float(var_y),
        "nwalkers": int(nwalkers),
        "burnin_steps": int(burnin_steps),
        "prod_steps": int(prod_steps),
        "N_POST_target": int(N_POST),
        "surrogate": surrogate_info,
        "counts": {
            "prod_steps_effective": int(nsteps_eff),
            "nwalkers_effective": int(nwalkers_eff),
            "N_total_unthinned": int(N_total),
            "thinned_samples_saved": thinned_count,
        },
        "diagnostics": {
            "window_W": int(W),
            "mean_logp_early": logp_early,
            "mean_logp_late": logp_late,
            "delta_logp_late_minus_early": logp_delta,
            "mean_b_early": b_early_list,
            "mean_b_late": b_late_list,
            "delta_mean_b_late_minus_early": b_delta_list,
            "rolling_blocks": block_stats,
            "split_rhat_per_dim": rhat_list,
            "split_rhat_max": rhat_max,
        },
        "ess": {
            "tau_per_dim": tau_list,
            "ess_per_dim": ess_list,
            "tau_max": tau_max,
            "ess_min": ess_min,
            "ess_mean": ess_mean,
            "thin_ess": thin_ess,
        },
        "timing_s": {
            "surrogate_setup_total": float(t_surr_total),
            "burnin": float(t_burn),
            "production": float(t_prod),
            "postproc": float(t_postproc),
            "save": float(t_save),
            "total_case": float(t_case),
        },
        "acceptance_fraction": {
            "burnin": {"mean": float(burn_acc_mean), "min": float(burn_acc_min), "max": float(burn_acc_max)},
            "production": {"mean": float(prod_acc_mean), "min": float(prod_acc_min), "max": float(prod_acc_max)},
        },
        "files": {
            "mcmc_samples": sample_path,
            "summary_json": "mcmc_summary.json",
            "surrogate_dir": os.path.relpath(surrogate_info["surrogate_dir"], out_dir)
                            if isinstance(surrogate_info.get("surrogate_dir", None), str) else None,
        },
        "data_used_for_surrogate": {
            "b_path": B_PATH,
            "y_path": Y_PATH,
            "rows_used": int(SURROGATE_NTRAIN),
            "b_slice": f"[:{SURROGATE_NTRAIN},:]",
            "y_slice": f"[:{SURROGATE_NTRAIN},:]",
        },
    }

    with open(os.path.join(out_dir, "mcmc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

print("\nALL DONE.")
