#!/usr/bin/env python3
"""Generate training data and test cases for the elliptic PDE example.

The log-permeability uses the sine basis from the experiment, and solutions
are evaluated at the ten verified scattered Grid 1 sensor locations.
"""

import argparse
import os
import math
import time
import hashlib
import numpy as np

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Generate the elliptic PDE dataset.")
parser.add_argument(
    "--output-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "elliptic_pde_results"),
    help="Directory receiving the data subdirectory and generated files.",
)
args = parser.parse_args()

# ----------------------------
# FEniCS
# ----------------------------
try:
    from dolfin import (
        UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, Function,
        DirichletBC, Constant, inner, grad, dx, Point, Expression,
        solve, set_log_level, LogLevel
    )
    FENICS_OK = True
except Exception as e:
    FENICS_OK = False
    FENICS_ERR = e

# ----------------------------
# User settings
# ----------------------------
savedir = os.path.abspath(args.output_dir)
data_dir = os.path.join(savedir, "data")
os.makedirs(savedir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

SEED = 42

# ----------------------------
# PDE / discretization settings
# ----------------------------
MESH_N = 32
PDE_DEGREE = 2
SOURCE_F = 1.0
QUIET_FENICS = True

# ----------------------------
# Basis and coefficient settings
#   k(x,y) = sum_{m=1..M_modes} sum_{l=1..L_modes} b_ml sin(2π m x) sin(2π l y)
# Here M_modes=L_modes=2 -> dim_u=4
# ----------------------------
M_modes = 2
L_modes = 2
dim_u = M_modes * L_modes
dim_v = 10

# ----------------------------
# Dataset generation settings
# ----------------------------
K_DATA = 20000
OVERWRITE_DATASET = True

# Testcases
N_TESTCASES = 2
OVERWRITE_TESTCASES = True

USE_DISTINCT_SEEDS_PER_TESTCASE = True
TESTCASE_SEED_STRIDE = 10007

# ----------------------------
# Fixed testcase selection
# ----------------------------
USE_FIXED_TESTCASES = True
FILL_REST_RANDOM = False            # if N_TESTCASES > len(fixed list), fill remaining with random prior draws

# Fixed coefficient vectors (length 4 for M=L=2)
b1 = np.array([-0.27029345,  0.77462187, -0.01628197, -0.4579129 ], dtype=np.float64)
b2 = np.array([ 0.39153082, -0.84271892, -0.74742908, -0.75483697], dtype=np.float64)
FIXED_B_LIST = [b1, b2]

# Known testcase-1 values at Grid 1, used as a regression check.
EXPECTED_TESTCASE_1_GRID1 = np.array(
    [0.0596921667, 0.0568632297, 0.0561620556, 0.0520923436,
     0.0706650093, 0.0410882719, 0.0413794853, 0.0655060112,
     0.0670271665, 0.0356916972],
    dtype=np.float32,
)

# ----------------------------
# Verified scattered Grid 1 sensors (rows 1--10)
# ----------------------------
GRID1_SENSOR_POINTS = np.array(
    [[0.3225806452, 0.6451612903],
     [0.7096774194, 0.3225806452],
     [0.6774193548, 0.6774193548],
     [0.3548387097, 0.2903225806],
     [0.5161290323, 0.4838709677],
     [0.4838709677, 0.8387096774],
     [0.1612903226, 0.4516129032],
     [0.4838709677, 0.6451612903],
     [0.3548387097, 0.4838709677],
     [0.8709677419, 0.5161290323]],
    dtype=np.float64,
)

# Use a Grid-1-specific filename so an old diagonal Grid-2 file named
# 10_sampling_locs_x_y.npy can never be loaded accidentally.
SENSOR_POINTS_PATH = os.path.join(data_dir, "10_sampling_locs_x_y_grid1.npy")

SOLUTION_GRID_NX = 32
SOLUTION_GRID_NY = 32
ON_GRID_TOL = 5e-10

PLOT_SENSOR_POINTS = True
SENSOR_PLOT_DPI = 250
SENSOR_PLOT_FIG = os.path.join(
    data_dir,
    f"10_sampling_locs_grid1_on_grid_{SOLUTION_GRID_NX}x{SOLUTION_GRID_NY}.png",
)

# ----------------------------
# Logging / timing settings
# ----------------------------
LOG_EVERY = 100
WARMUP_N = 10
TIME_PROBE = True
N_PROBE = 25

# ----------------------------
# Prior for coefficients b_ml ~ N(0, 1/(m+l))
# Ordering used in code:
#   idx 0: (m=1,l=1)
#   idx 1: (m=1,l=2)
#   idx 2: (m=2,l=1)
#   idx 3: (m=2,l=2)
# ----------------------------
prior_mu = np.zeros(dim_u, dtype=np.float64)
prior_vars = []
for m in range(1, M_modes + 1):
    for l in range(1, L_modes + 1):
        prior_vars.append(1.0 / (m + l))
prior_cov = np.diag(prior_vars).astype(np.float64)

# ----------------------------
# Utilities: grid validation + plotting
# ----------------------------
def solution_grid_nodes(nx: int, ny: int):
    gx = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    gy = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    return gx, gy

def points_on_grid(points_xy: np.ndarray, gx: np.ndarray, gy: np.ndarray, tol: float):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    on = []
    for (x, y) in pts:
        okx = np.min(np.abs(gx - x)) <= tol
        oky = np.min(np.abs(gy - y)) <= tol
        on.append(bool(okx and oky))
    return np.array(on, dtype=bool)

def plot_sensor_points_on_grid(points_xy: np.ndarray, gx: np.ndarray, gy: np.ndarray, out_png: str):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)

    GX, GY = np.meshgrid(gx, gy)  # both (ny, nx)
    grid_xy = np.stack([GX.ravel(), GY.ravel()], axis=1)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=8, alpha=0.35, linewidths=0)

    ax.scatter(pts[:, 0], pts[:, 1], s=80, facecolors="none",
              edgecolors="red", linewidths=2, zorder=3)

    for i, (x, y) in enumerate(pts, start=1):
        ax.text(x + 0.012, y + 0.012, str(i), fontsize=11,
                color="black", weight="bold", zorder=4)

    ax.set_title(f"Sensor locations on {len(gx)}x{len(gy)} grid", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=SENSOR_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[sensor] saved sensor grid plot: {out_png}")

# ----------------------------
# Sensor points: Grid 1 only
# ----------------------------
def configure_grid1_sensor_points(dim_v: int) -> np.ndarray:
    pts = GRID1_SENSOR_POINTS.copy()
    if pts.shape != (dim_v, 2):
        raise ValueError(
            f"GRID1_SENSOR_POINTS has shape {pts.shape}, expected {(dim_v, 2)}"
        )

    gx, gy = solution_grid_nodes(SOLUTION_GRID_NX, SOLUTION_GRID_NY)
    on_grid = points_on_grid(pts, gx, gy, tol=ON_GRID_TOL)
    if not np.all(on_grid):
        bad_rows = (np.flatnonzero(~on_grid) + 1).tolist()
        raise RuntimeError(
            f"Grid 1 sensor rows {bad_rows} are not on the "
            f"{SOLUTION_GRID_NX}x{SOLUTION_GRID_NY} grid"
        )

    # Always write the verified values. Never load the legacy Grid-2 file.
    np.save(SENSOR_POINTS_PATH, pts)
    print(f"[sensor] using verified scattered Grid 1 sensors (rows 1--10)")
    print(pts)
    print(f"[sensor] saved Grid 1 sensors: {SENSOR_POINTS_PATH} shape={pts.shape}")

    if PLOT_SENSOR_POINTS:
        plot_sensor_points_on_grid(pts, gx, gy, SENSOR_PLOT_FIG)

    return pts

# ----------------------------
# FEniCS PDE forward map (returns ONLY 10 sensor values)
# ----------------------------
if not FENICS_OK:
    raise RuntimeError(
        "FEniCS/dolfin import failed in this environment. "
        f"Error: {type(FENICS_ERR).__name__}: {FENICS_ERR}"
    )

if QUIET_FENICS:
    try:
        set_log_level(LogLevel.ERROR)
    except Exception:
        pass

sensor_points = configure_grid1_sensor_points(dim_v=dim_v)

mesh = UnitSquareMesh(MESH_N, MESH_N)
V = FunctionSpace(mesh, "P", PDE_DEGREE)

u = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, Constant(0.0), "on_boundary")
f = Constant(float(SOURCE_F))

def fenics_forward_sensors(b_1d: np.ndarray) -> np.ndarray:
    """
    Uses:
      log k(x,y) = sum_{m=1..M_modes} sum_{l=1..L_modes} b_ml sin(2π m x) sin(2π l y)
      k(x,y) = exp(log k(x,y))
    """
    b_1d = np.asarray(b_1d, dtype=np.float64).reshape(dim_u)

    # For M=L=2, unpack in the ordering:
    # [b11, b12, b21, b22] where first index is m, second is l
    b11, b12, b21, b22 = [float(x) for x in b_1d.tolist()]

    # Paper basis: sin(2π m x) sin(2π l y)
    logk_expr = Expression(
        "b11*sin(2*pi*1*x[0])*sin(2*pi*1*x[1])"
        " + b12*sin(2*pi*1*x[0])*sin(2*pi*2*x[1])"
        " + b21*sin(2*pi*2*x[0])*sin(2*pi*1*x[1])"
        " + b22*sin(2*pi*2*x[0])*sin(2*pi*2*x[1])",
        degree=4,
        pi=math.pi,
        b11=b11, b12=b12, b21=b21, b22=b22
    )

    k_expr = Expression("exp(lk)", degree=4, lk=logk_expr)

    a_form = inner(k_expr * grad(u), grad(v)) * dx
    L_form = f * v * dx

    uh = Function(V)
    solve(a_form == L_form, uh, bc)

    y = np.empty((dim_v,), dtype=np.float64)
    for i in range(dim_v):
        px, py = float(sensor_points[i, 0]), float(sensor_points[i, 1])
        y[i] = float(uh(Point(px, py)))
    return y

# ----------------------------
# Pretty time helper
# ----------------------------
def fmt_seconds(sec: float) -> str:
    sec = float(sec)
    if not np.isfinite(sec):
        return "nan"
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m:d}m{s:04.1f}s"
    h = int(m // 60)
    m = m - 60 * h
    return f"{h:d}h{m:02d}m{s:04.1f}s"

# ----------------------------
# Short timing probe
# ----------------------------
def timing_probe(n_probe: int):
    rng_probe = np.random.default_rng(SEED + 999)

    for _ in range(2):
        b = rng_probe.multivariate_normal(mean=prior_mu, cov=prior_cov)
        _ = fenics_forward_sensors(b)

    t0 = time.perf_counter()
    for _ in range(n_probe):
        b = rng_probe.multivariate_normal(mean=prior_mu, cov=prior_cov)
        _ = fenics_forward_sensors(b)
    t1 = time.perf_counter()

    t_per = (t1 - t0) / max(1, n_probe)
    est_total = t_per * K_DATA
    print("\n[probe] Timing probe:")
    print(f"  n_probe={n_probe}")
    print(f"  avg time per sample ≈ {t_per:.6f} s")
    print(f"  est time for K_DATA={K_DATA} ≈ {fmt_seconds(est_total)}")
    print("")

# ----------------------------
# Dataset generation (with logging)
# ----------------------------
def generate_dataset(K: int):
    x_out = os.path.join(data_dir, f"b_mn_samples_{K}.npy")
    y_out = os.path.join(data_dir, f"sampled_solutions_{K}_10_locs.npy")

    if (not OVERWRITE_DATASET) and os.path.exists(x_out) and os.path.exists(y_out):
        print("\n[data] Dataset files already exist; skipping generation:")
        print(f"  {x_out}")
        print(f"  {y_out}")
        return

    print("\n========== DATA GENERATION (FEniCS) ==========")
    print(f"[data] FE space: P{PDE_DEGREE}  | mesh: UnitSquareMesh({MESH_N},{MESH_N})")
    print(f"[data] basis: sin(2π m x) sin(2π l y), M={M_modes}, L={L_modes} -> dim_u={dim_u}")
    print(f"[data] sensor points saved at: {SENSOR_POINTS_PATH}")
    if PLOT_SENSOR_POINTS:
        print(f"[data] sensor grid plot saved at: {SENSOR_PLOT_FIG}")
    print(f"[data] Generating K={K} pairs (b, y_at_10_sensors) ...")
    print("=============================================\n")

    rng_data = np.random.default_rng(SEED + 777)

    X = np.empty((K, dim_u), dtype=np.float64)
    Y = np.empty((K, dim_v), dtype=np.float64)

    t0 = time.perf_counter()
    solve_times = []

    for k in range(K):
        b = rng_data.multivariate_normal(mean=prior_mu, cov=prior_cov)

        t_s0 = time.perf_counter()
        y = fenics_forward_sensors(b)
        t_s1 = time.perf_counter()

        X[k, :] = b
        Y[k, :] = y

        solve_times.append(t_s1 - t_s0)

        done = k + 1
        if (done % LOG_EVERY == 0) or (done == K):
            now = time.perf_counter()
            elapsed = now - t0

            avg = elapsed / done
            m = min(len(solve_times), LOG_EVERY)
            roll = float(np.mean(solve_times[-m:]))

            if done >= WARMUP_N:
                eta = avg * (K - done)
                eta_str = fmt_seconds(eta)
            else:
                eta_str = "warming up..."

            rate = done / max(elapsed, 1e-12)

            print(
                f"[data] {done:6d}/{K} | elapsed {fmt_seconds(elapsed)} "
                f"| avg {avg:.4f}s/sample | roll{m} {roll:.4f}s/sample "
                f"| {rate:.2f} samples/s | ETA {eta_str}"
            )

    t1 = time.perf_counter()
    np.save(x_out, X.astype(np.float32))
    np.save(y_out, Y.astype(np.float32))

    total = t1 - t0
    print("\n[saved] Dataset files:")
    print(f"  {x_out} shape={X.shape}")
    print(f"  {y_out} shape={Y.shape}")
    print("[data] Final timing summary:")
    print(f"  total wall time: {fmt_seconds(total)}")
    print(f"  avg time/sample: {total / K:.6f} s")
    if len(solve_times) > 0:
        st = np.asarray(solve_times, dtype=np.float64)
        print(f"  solve-time stats: mean={st.mean():.6f}s  median={np.median(st):.6f}s  p95={np.quantile(st,0.95):.6f}s")
    print("=============================================\n")

# ----------------------------
# Testcase generation
# ----------------------------
def generate_testcases(n_cases: int):
    all_exist = True
    for i in range(1, n_cases + 1):
        y_path = os.path.join(data_dir, f"testcase_{i}.npy")
        b_path = os.path.join(data_dir, f"testcase_{i}_bmn.npy")
        if not (os.path.exists(y_path) and os.path.exists(b_path)):
            all_exist = False
            break

    if (not OVERWRITE_TESTCASES) and all_exist:
        print("\n[test] All testcase files already exist; skipping generation.")
        return

    print("\n========== TESTCASE GENERATION (FEniCS) ==========")
    print(f"[test] Creating {n_cases} testcases.")
    print(f"[test] USE_FIXED_TESTCASES={USE_FIXED_TESTCASES} | fixed_count={len(FIXED_B_LIST)} | fill_rest_random={FILL_REST_RANDOM}")
    print(f"[test] distinct_seeds_per_testcase={USE_DISTINCT_SEEDS_PER_TESTCASE}  stride={TESTCASE_SEED_STRIDE}")
    print("===============================================\n")

    t0_all = time.perf_counter()
    rng_seq = np.random.default_rng(SEED + 2026)

    for i in range(1, n_cases + 1):
        tc_y_path = os.path.join(data_dir, f"testcase_{i}.npy")
        tc_b_path = os.path.join(data_dir, f"testcase_{i}_bmn.npy")

        if (not OVERWRITE_TESTCASES) and os.path.exists(tc_y_path) and os.path.exists(tc_b_path):
            print(f"[test] exists; skipping testcase_{i}")
            continue

        # ---- choose b ----
        used_fixed = False
        if USE_FIXED_TESTCASES and (i <= len(FIXED_B_LIST)):
            b = np.asarray(FIXED_B_LIST[i - 1], dtype=np.float64).reshape(dim_u)
            used_fixed = True
            tc_seed = None
        else:
            if USE_FIXED_TESTCASES and (not FILL_REST_RANDOM):
                raise RuntimeError(
                    f"Requested N_TESTCASES={n_cases} but only {len(FIXED_B_LIST)} fixed b vectors provided "
                    f"and FILL_REST_RANDOM=False. Either add more fixed vectors or enable FILL_REST_RANDOM."
                )

            if USE_DISTINCT_SEEDS_PER_TESTCASE:
                tc_seed = SEED + TESTCASE_SEED_STRIDE * i
                rng_tc = np.random.default_rng(tc_seed)
            else:
                tc_seed = None
                rng_tc = rng_seq

            b = rng_tc.multivariate_normal(mean=prior_mu, cov=prior_cov).astype(np.float64)

        # ---- solve ----
        t0 = time.perf_counter()
        y = fenics_forward_sensors(b)
        t1 = time.perf_counter()

        # Fail immediately if fixed testcase 1 is not being sampled at the
        # verified Grid 1 coordinates in the correct row order.
        if used_fixed and i == 1:
            y32 = y.astype(np.float32)
            error = y32 - EXPECTED_TESTCASE_1_GRID1
            max_abs_error = float(np.max(np.abs(error)))
            matches = bool(np.allclose(
                y32, EXPECTED_TESTCASE_1_GRID1, rtol=0.0, atol=5.0e-7
            ))
            print("[test] testcase 1 Grid 1 regression check:")
            print(f"  computed: {y32}")
            print(f"  expected: {EXPECTED_TESTCASE_1_GRID1}")
            print(f"  max_abs_error={max_abs_error:.12e}  matches={matches}")
            if not matches:
                raise RuntimeError(
                    "Testcase 1 does not match the verified Grid 1 values; "
                    "refusing to save data with an incorrect sensor mapping."
                )

        np.save(tc_b_path, b.astype(np.float32))
        np.save(tc_y_path, y.astype(np.float32))

        y_md5 = hashlib.md5(y.astype(np.float64).tobytes()).hexdigest()
        if used_fixed:
            seed_str = "FIXED"
        else:
            seed_str = str(tc_seed) if (tc_seed is not None) else "SEQUENTIAL"

        print(f"[saved] testcase_{i}: mode={seed_str} | md5(y)={y_md5} | solve {fmt_seconds(t1-t0)}")

    t1_all = time.perf_counter()
    print("\n[test] Done generating testcases.")
    print(f"[test] total testcase wall time: {fmt_seconds(t1_all - t0_all)}")
    print("===============================================\n")

# ----------------------------
# Main
# ----------------------------
print("\n========== RUN CONFIG ==========")
print(f"[VERIFY] savedir (abs):  {os.path.abspath(savedir)}")
print(f"[VERIFY] data_dir (abs): {os.path.abspath(data_dir)}")
print(f"[VERIFY] SEED={SEED}")
print(f"[VERIFY] dim_u={dim_u} dim_v={dim_v}")
print(f"[VERIFY] BASIS: sin(2π m x) sin(2π l y) with M={M_modes}, L={L_modes}")
print(f"[VERIFY] prior vars (ordered b11,b12,b21,b22): {prior_vars}")
print(f"[VERIFY] FE space: P{PDE_DEGREE}")
print(f"[VERIFY] mesh: UnitSquareMesh({MESH_N},{MESH_N})")
print(f"[VERIFY] sensor set: verified scattered Grid 1 (rows 1--10)")
print(f"[VERIFY] sensor_points file: {SENSOR_POINTS_PATH}")
print(f"[VERIFY] sensor validation grid: {SOLUTION_GRID_NX}x{SOLUTION_GRID_NY}")
print(f"[VERIFY] sensor_points:\n{sensor_points}")
print(f"[VERIFY] sensor plot: enabled={PLOT_SENSOR_POINTS} out={SENSOR_PLOT_FIG}")
print(f"[VERIFY] K_DATA={K_DATA} OVERWRITE_DATASET={OVERWRITE_DATASET}")
print(f"[VERIFY] N_TESTCASES={N_TESTCASES} OVERWRITE_TESTCASES={OVERWRITE_TESTCASES}")
print(f"[VERIFY] USE_FIXED_TESTCASES={USE_FIXED_TESTCASES} fixed_count={len(FIXED_B_LIST)} fill_rest_random={FILL_REST_RANDOM}")
print(f"[VERIFY] distinct_seeds_per_testcase={USE_DISTINCT_SEEDS_PER_TESTCASE} stride={TESTCASE_SEED_STRIDE}")
print(f"[VERIFY] LOG_EVERY={LOG_EVERY} TIME_PROBE={TIME_PROBE} N_PROBE={N_PROBE}")
print("================================\n")

generate_testcases(N_TESTCASES)
if TIME_PROBE:
    timing_probe(N_PROBE)
generate_dataset(K_DATA)
