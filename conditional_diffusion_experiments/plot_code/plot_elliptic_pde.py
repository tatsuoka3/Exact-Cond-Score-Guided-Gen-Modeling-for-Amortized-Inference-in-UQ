#!/usr/bin/env python3
"""Create the elliptic PDE figures from saved experiment results.

The plotting notebook supplied with the experiment contained separate,
duplicated blocks for the two test cases. This script keeps the same intended
figures and filenames while sharing all loading, PDE-solving, and plotting
code. NN and thinned-MCMC coefficient samples are passed through the FEniCS
forward solver before solution-error plots are constructed. All NN solution
fields are saved once so Figure 5 sample indices can later be changed without
another PDE solve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


mesh_size = 32
pde_degree = 2
source_value = 1.0
sample_limit = 5000
coefficient_count = 4
sensor_count = 10

variance_u = 1.0e-2
variance_v = 1.0e-5
variance_y = 1.0e-6
observation_noise_std = 0.0025

coefficient_labels = [r"$b_{11}$", r"$b_{12}$", r"$b_{21}$", r"$b_{22}$"]
generated_field_indices = {
    1: (102, 416, 2889),
    2: (1500, 63, 893),
}


@dataclass
class CaseData:
    testcase: int
    true_coefficients: np.ndarray
    true_sensors: np.ndarray
    true_field: np.ndarray
    prior_coefficients: np.ndarray
    prior_sensors: np.ndarray
    nn_coefficients: np.ndarray
    nn_sensors: np.ndarray
    mcmc_coefficients: np.ndarray
    mcmc_sensors: np.ndarray
    field_indices: np.ndarray
    nn_fields: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the elliptic PDE example.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "elliptic_pde_results",
        help="Elliptic PDE result directory containing data and sample files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory receiving figures; defaults to INPUT_DIR/figures.",
    )
    parser.add_argument(
        "--solve-missing",
        action="store_true",
        help="Solve only plot inputs that have not already been saved.",
    )
    return parser.parse_args()


def load_array(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    array = np.load(path)
    print(f"[load] {label}: {array.shape} <- {path}", flush=True)
    return array


def as_coefficients(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1 and values.size == coefficient_count:
        values = values.reshape(1, coefficient_count)
    elif values.ndim == 3 and values.shape[1:] == (2, 2):
        values = values.reshape(values.shape[0], coefficient_count)
    if values.ndim != 2 or values.shape[1] < coefficient_count:
        raise ValueError(f"{label} must have shape (N, 4); got {values.shape}")
    return np.asarray(values[:, :coefficient_count], dtype=np.float64)


def as_observations(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != sensor_count:
        raise ValueError(f"{label} must have shape (N, 10); got {values.shape}")
    return values


def as_solution_field(array: np.ndarray, label: str) -> np.ndarray:
    values = np.squeeze(np.asarray(array, dtype=np.float64))
    if values.shape != (mesh_size, mesh_size):
        raise ValueError(
            f"{label} must have shape ({mesh_size}, {mesh_size}); got {values.shape}"
        )
    return values


def nn_sample_path(input_dir: Path, testcase: int) -> Path:
    name = (
        f"NN_output_testcase_{testcase}_VAR_Y_1e-06_VAR_U_0.01_"
        "VAR_V_1e-05_Ntrain_5000_"
        "yStd_0.0025_FNw128_FNlayers1_K_20000.npy"
    )
    preferred = input_dir / name
    if preferred.is_file():
        return preferred
    matches = sorted(input_dir.glob(f"NN_output_testcase_{testcase}_*.npy"))
    if len(matches) == 1:
        return matches[0]
    return preferred


def coefficient_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(0.0, 1.0, mesh_size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    basis = np.stack(
        [
            np.sin(2.0 * np.pi * m * x_grid)
            * np.sin(2.0 * np.pi * n * y_grid)
            for m in range(1, 3)
            for n in range(1, 3)
        ]
    )
    return x_grid, y_grid, basis


def log_permeability(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = as_coefficients(coefficients, "coefficients")
    return np.einsum("nk,kij->nij", values, basis)


def permeability(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.exp(log_permeability(coefficients, basis))


class EllipticPDESolver:
    """FEniCS forward solver matching generate_elliptic_pde_data.py."""

    def __init__(self, sensor_points: np.ndarray):
        try:
            from dolfin import (
                Constant,
                DirichletBC,
                Expression,
                Function,
                FunctionSpace,
                LogLevel,
                Point,
                TestFunction,
                TrialFunction,
                UnitSquareMesh,
                dx,
                grad,
                inner,
                set_log_level,
                solve,
            )
        except Exception as error:
            raise RuntimeError(
                "The elliptic PDE plotting job requires FEniCS/dolfin"
            ) from error

        self.Constant = Constant
        self.Expression = Expression
        self.Function = Function
        self.Point = Point
        self.dx = dx
        self.grad = grad
        self.inner = inner
        self.solve_equation = solve
        self.sensor_points = np.asarray(sensor_points, dtype=np.float64)

        try:
            set_log_level(LogLevel.ERROR)
        except Exception:
            pass

        mesh = UnitSquareMesh(mesh_size, mesh_size)
        self.space = FunctionSpace(mesh, "P", pde_degree)
        self.trial = TrialFunction(self.space)
        self.test = TestFunction(self.space)
        self.boundary = DirichletBC(self.space, Constant(0.0), "on_boundary")
        self.source = Constant(source_value)

    def solve(self, coefficients: np.ndarray, include_field: bool = False):
        b11, b12, b21, b22 = np.asarray(coefficients, dtype=float).reshape(4)
        log_k = self.Expression(
            "b11*sin(2*pi*1*x[0])*sin(2*pi*1*x[1])"
            " + b12*sin(2*pi*1*x[0])*sin(2*pi*2*x[1])"
            " + b21*sin(2*pi*2*x[0])*sin(2*pi*1*x[1])"
            " + b22*sin(2*pi*2*x[0])*sin(2*pi*2*x[1])",
            degree=4,
            pi=np.pi,
            b11=float(b11),
            b12=float(b12),
            b21=float(b21),
            b22=float(b22),
        )
        kappa = self.Expression("exp(lk)", degree=4, lk=log_k)
        lhs = self.inner(kappa * self.grad(self.trial), self.grad(self.test)) * self.dx
        rhs = self.source * self.test * self.dx
        solution = self.Function(self.space)
        self.solve_equation(lhs == rhs, solution, self.boundary)

        sensors = np.array(
            [
                float(solution(self.Point(float(x), float(y))))
                for x, y in self.sensor_points
            ],
            dtype=np.float64,
        )
        if not include_field:
            return sensors, None

        coordinates = np.linspace(0.0, 1.0, mesh_size)
        field = np.empty((mesh_size, mesh_size), dtype=np.float64)
        for row, y in enumerate(coordinates):
            for column, x in enumerate(coordinates):
                field[row, column] = float(solution(self.Point(float(x), float(y))))
        return sensors, field

    def solve_samples(
        self,
        coefficients: np.ndarray,
        include_fields: bool = False,
        label: str = "samples",
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = as_coefficients(coefficients, label)
        sensors = np.empty((coefficients.shape[0], sensor_count), dtype=np.float64)
        fields = (
            np.empty(
                (coefficients.shape[0], mesh_size, mesh_size),
                dtype=np.float64,
            )
            if include_fields
            else np.empty((0, mesh_size, mesh_size), dtype=np.float64)
        )

        for index, values in enumerate(coefficients):
            sensor_values, field = self.solve(values, include_field=include_fields)
            sensors[index] = sensor_values
            if include_fields:
                fields[index] = field
            if (index + 1) % 100 == 0 or index + 1 == coefficients.shape[0]:
                print(
                    f"[solve] {label}: {index + 1}/{coefficients.shape[0]}",
                    flush=True,
                )

        return sensors, fields


def closest_field_index(
    coefficients: np.ndarray,
    true_coefficients: np.ndarray,
    basis: np.ndarray,
) -> int:
    truth = log_permeability(true_coefficients, basis)[0]
    best_index = 0
    best_distance = np.inf
    for start in range(0, coefficients.shape[0], 512):
        stop = min(start + 512, coefficients.shape[0])
        fields = log_permeability(coefficients[start:stop], basis)
        distances = np.linalg.norm(
            (fields - truth).reshape(stop - start, -1),
            axis=1,
        )
        local_index = int(np.argmin(distances))
        if distances[local_index] < best_distance:
            best_distance = float(distances[local_index])
            best_index = start + local_index
    return best_index


def prepare_case(
    testcase: int,
    input_dir: Path,
    solved_dir: Path,
    prior_coefficients: np.ndarray,
    prior_sensors: np.ndarray,
    sensor_points: np.ndarray,
    basis: np.ndarray,
    solve_missing: bool,
) -> CaseData:
    data_dir = input_dir / "data"
    true_coefficients = as_coefficients(
        load_array(data_dir / f"testcase_{testcase}_bmn.npy", "true coefficients"),
        "true coefficients",
    )[0]
    saved_true_sensors = as_observations(
        load_array(data_dir / f"testcase_{testcase}.npy", "true observations"),
        "true observations",
    )[0]
    nn_coefficients = as_coefficients(
        load_array(nn_sample_path(input_dir, testcase), "NN coefficients"),
        "NN coefficients",
    )[:sample_limit]
    mcmc_coefficients = as_coefficients(
        load_array(
            input_dir / f"MCMC_samples_testcase{testcase}.npy",
            "thinned MCMC coefficients",
        ),
        "thinned MCMC coefficients",
    )[:sample_limit]

    closest = closest_field_index(nn_coefficients, true_coefficients, basis)
    selected = (closest, *generated_field_indices[testcase])
    if max(selected) >= nn_coefficients.shape[0]:
        raise IndexError(
            f"Figure 5 requires NN index {max(selected)}, but only "
            f"{nn_coefficients.shape[0]} samples were saved"
        )

    true_field_path = solved_dir / f"true_solution_field_testcase{testcase}.npy"
    nn_sensors_path = solved_dir / f"NN_solution_sensors_testcase{testcase}.npy"
    nn_fields_path = solved_dir / f"NN_solution_fields_all_testcase{testcase}.npy"
    mcmc_sensors_path = solved_dir / f"MCMC_solution_sensors_testcase{testcase}.npy"

    solver: EllipticPDESolver | None = None

    def get_solver(label: str) -> EllipticPDESolver:
        nonlocal solver
        if not solve_missing:
            raise FileNotFoundError(
                f"Missing solved plot input: {label}. Run once with --solve-missing."
            )
        if solver is None:
            solver = EllipticPDESolver(sensor_points)
        return solver

    if true_field_path.is_file():
        true_field = as_solution_field(
            load_array(true_field_path, "saved true solution field"),
            "saved true solution field",
        )
    else:
        solved_true_sensors, true_field = get_solver("true solution field").solve(
            true_coefficients,
            include_field=True,
        )
        discrepancy = np.max(np.abs(solved_true_sensors - saved_true_sensors))
        print(
            f"[check] testcase {testcase} saved/solved sensor max error: "
            f"{discrepancy:.6e}",
            flush=True,
        )
        np.save(true_field_path, true_field)

    computed_nn_sensors = None
    if nn_fields_path.is_file():
        all_nn_fields = np.asarray(
            load_array(nn_fields_path, "saved NN solution fields"),
            dtype=np.float64,
        )
        expected_shape = (nn_coefficients.shape[0], mesh_size, mesh_size)
        if all_nn_fields.shape != expected_shape:
            raise ValueError(
                f"Saved NN fields must have shape {expected_shape}; "
                f"got {all_nn_fields.shape}"
            )
        solved_all_nn_fields = False
    else:
        computed_nn_sensors, all_nn_fields = get_solver(
            "all NN solution fields"
        ).solve_samples(
            nn_coefficients,
            include_fields=True,
            label=f"NN testcase {testcase}",
        )
        np.save(nn_fields_path, all_nn_fields)
        solved_all_nn_fields = True

    if nn_sensors_path.is_file():
        nn_sensors = as_observations(
            load_array(nn_sensors_path, "saved NN solution sensors"),
            "saved NN solution sensors",
        )
        if nn_sensors.shape[0] != nn_coefficients.shape[0]:
            raise ValueError("Saved NN sensor count does not match NN coefficient count")
    elif computed_nn_sensors is not None:
        nn_sensors = computed_nn_sensors
        np.save(nn_sensors_path, nn_sensors)
    else:
        nn_sensors, _ = get_solver("NN solution sensors").solve_samples(
            nn_coefficients,
            label=f"NN testcase {testcase}",
        )
        np.save(nn_sensors_path, nn_sensors)

    if mcmc_sensors_path.is_file():
        mcmc_sensors = as_observations(
            load_array(mcmc_sensors_path, "saved MCMC solution sensors"),
            "saved MCMC solution sensors",
        )
        if mcmc_sensors.shape[0] != mcmc_coefficients.shape[0]:
            raise ValueError("Saved MCMC sensor count does not match MCMC coefficient count")
    else:
        mcmc_sensors, _ = get_solver("MCMC solution sensors").solve_samples(
            mcmc_coefficients,
            label=f"MCMC testcase {testcase}",
        )
        np.save(mcmc_sensors_path, mcmc_sensors)

    nn_fields = all_nn_fields[np.asarray(selected, dtype=int)]

    print(
        f"[ready] testcase {testcase}: all NN solution fields available; "
        f"generated now={solved_all_nn_fields}",
        flush=True,
    )

    return CaseData(
        testcase=testcase,
        true_coefficients=true_coefficients,
        true_sensors=saved_true_sensors,
        true_field=np.asarray(true_field),
        prior_coefficients=prior_coefficients,
        prior_sensors=prior_sensors,
        nn_coefficients=nn_coefficients,
        nn_sensors=nn_sensors,
        mcmc_coefficients=mcmc_coefficients,
        mcmc_sensors=mcmc_sensors,
        field_indices=np.asarray(selected, dtype=int),
        nn_fields=nn_fields,
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[saved] {pdf_path}", flush=True)
    print(f"[saved] {png_path}", flush=True)


def plot_fields(
    case: CaseData,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    basis: np.ndarray,
    sensor_points: np.ndarray,
    output_dir: Path,
) -> None:
    selected_coefficients = case.nn_coefficients[case.field_indices]
    permeability_fields = np.concatenate(
        [
            permeability(case.true_coefficients, basis),
            permeability(selected_coefficients, basis),
        ],
        axis=0,
    )
    solution_fields = np.concatenate(
        [case.true_field[None, :, :], case.nn_fields],
        axis=0,
    )
    solution_fields = np.clip(
        np.nan_to_num(solution_fields, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )

    permeability_titles = [
        "True permeability field",
        "Closest sample",
        "Generated sample",
        "Generated sample",
        "Generated sample",
    ]
    solution_titles = [
        "True solution field",
        "Closest sample",
        "Generated sample",
        "Generated sample",
        "Generated sample",
    ]
    figure, all_axes = plt.subplots(
        2,
        6,
        figsize=(19, 7.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.045]},
    )
    axes = all_axes[:, :5]
    permeability_colorbar_axis = all_axes[0, 5]
    solution_colorbar_axis = all_axes[1, 5]

    permeability_levels = np.linspace(
        float(np.min(permeability_fields)),
        float(np.max(permeability_fields)),
        51,
    )
    solution_levels = np.linspace(
        float(np.min(solution_fields)),
        float(np.max(solution_fields)),
        21,
    )

    permeability_contour = None
    solution_contour = None
    for column in range(5):
        top = axes[0, column]
        permeability_contour = top.contourf(
            x_grid,
            y_grid,
            permeability_fields[column],
            levels=permeability_levels,
            cmap="coolwarm",
        )
        top.set_title(permeability_titles[column], fontsize=11)
        top.set_aspect("equal")
        top.set_xlim(0.0, 1.0)
        top.set_ylim(0.0, 1.0)
        top.tick_params(labelsize=9)
        if column == 0:
            top.set_ylabel("y")
        else:
            top.tick_params(labelleft=False)
        top.tick_params(labelbottom=False)

        bottom = axes[1, column]
        solution_contour = bottom.contourf(
            x_grid,
            y_grid,
            solution_fields[column],
            levels=solution_levels,
            cmap="viridis",
        )
        bottom.set_title(solution_titles[column], fontsize=11)
        bottom.scatter(
            sensor_points[:, 0],
            sensor_points[:, 1],
            facecolors="none",
            edgecolors="red",
            s=80,
            linewidth=2.8,
        )
        if column == 0:
            for index, (x, y) in enumerate(sensor_points, start=1):
                bottom.text(
                    x - 0.035,
                    y - 0.085,
                    str(index),
                    fontsize=14,
                    fontweight="bold",
                )
        bottom.set_aspect("equal")
        bottom.set_xlim(0.0, 1.0)
        bottom.set_ylim(0.0, 1.0)
        bottom.set_xlabel("x")
        bottom.tick_params(labelsize=9)
        if column == 0:
            bottom.set_ylabel("y")
        else:
            bottom.tick_params(labelleft=False)

    figure.colorbar(
        permeability_contour,
        cax=permeability_colorbar_axis,
    ).set_label("K(x, y)")
    figure.colorbar(
        solution_contour,
        cax=solution_colorbar_axis,
    ).set_label("u(x, y)")
    save_figure(figure, output_dir, "fig5_a" if case.testcase == 1 else "fig5_b")


def signed_relative_errors(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(truth) > 1.0e-12, truth, 1.0e-12)
    return (samples - truth[None, :]) / denominator[None, :]


def plot_combined_sensor_errors(
    cases: dict[int, CaseData],
    output_dir: Path,
) -> None:
    selected_sensors = (0, 5, 8)
    methods = (
        ("Prior", "prior_sensors"),
        ("Posterior", "nn_sensors"),
        ("MCMC Posterior", "mcmc_sensors"),
    )
    errors: dict[tuple[int, str], np.ndarray] = {}
    for testcase, case in cases.items():
        for label, attribute in methods:
            errors[(testcase, label)] = signed_relative_errors(
                getattr(case, attribute),
                case.true_sensors,
            )

    figure, all_axes = plt.subplots(
        3,
        7,
        figsize=(26, 11),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.1, 1, 1, 1]},
    )

    for row in range(3):
        figure.delaxes(all_axes[row, 3])

    for block, testcase in enumerate((1, 2)):
        column_offset = 0 if block == 0 else 4
        for local_column, sensor_index in enumerate(selected_sensors):
            for row, (label, _) in enumerate(methods):
                color = {
                    "Prior": "blue",
                    "Posterior": "magenta" if testcase == 1 else "red",
                    "MCMC Posterior": "green",
                }[label]
                axis = all_axes[row, column_offset + local_column]
                axis.hist(
                    errors[(testcase, label)][:, sensor_index],
                    bins=50,
                    density=True,
                    color=color,
                    histtype="stepfilled",
                )
                axis.set_xlim(-1.0, 1.0)
                axis.set_ylim(0.0, 15.0)
                axis.set_title(
                    f"{label} Error, Location {sensor_index + 1}",
                    fontsize=12,
                )
                axis.tick_params(labelsize=14)
                axis.grid(which="both", linestyle="--", linewidth=0.5)
                if local_column == 0:
                    axis.set_ylabel("Density", fontsize=16)
                if row == 2:
                    axis.set_xlabel("Signed relative error", fontsize=16)

    legend_labels = (
        ("Prior Density", "blue"),
        ("Posterior Density (Test case 1)", "magenta"),
        ("Posterior Density (Test case 2)", "red"),
        ("MCMC Posterior Density", "green"),
    )
    legend = [
        Line2D([0], [0], color=color, linewidth=8, label=label)
        for label, color in legend_labels
    ]
    figure.legend(
        legend,
        [item.get_label() for item in legend],
        loc="lower center",
        ncol=4,
        fontsize=16,
        handlelength=2.2,
        columnspacing=1.8,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.13, wspace=0.55, hspace=0.48)
    save_figure(figure, output_dir, "fig6_pde_combined")


def permeability_relative_mse(
    coefficients: np.ndarray,
    true_coefficients: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    truth = permeability(true_coefficients, basis)[0]
    denominator = float(np.mean(truth**2))
    result = np.empty(coefficients.shape[0], dtype=np.float64)
    for start in range(0, coefficients.shape[0], 512):
        stop = min(start + 512, coefficients.shape[0])
        fields = permeability(coefficients[start:stop], basis)
        result[start:stop] = np.mean(
            (fields - truth[None, :, :]) ** 2,
            axis=(1, 2),
        ) / denominator
    return result


def solution_relative_mse(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean((samples - truth[None, :]) ** 2, axis=1) / np.mean(truth**2)


def plot_relative_mse(
    case: CaseData,
    basis: np.ndarray,
    output_dir: Path,
) -> None:
    methods = (
        ("Prior", case.prior_coefficients, case.prior_sensors, "red"),
        (
            r"NN: $\sigma_Y^2=10^{-6}$",
            case.nn_coefficients,
            case.nn_sensors,
            "magenta",
        ),
        ("MCMC", case.mcmc_coefficients, case.mcmc_sensors, "green"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (label, coefficients, sensors, color) in zip(axes, methods):
        x_values = solution_relative_mse(sensors, case.true_sensors)
        y_values = permeability_relative_mse(
            coefficients,
            case.true_coefficients,
            basis,
        )
        axis.scatter(x_values, y_values, color=color, alpha=0.25, s=7)
        axis.set_title(f"Test case {case.testcase}: {label}", fontsize=14)
        axis.set_xlim(0.0, 0.1)
        axis.set_ylim(0.0, 2.0)
        axis.set_xlabel("Solution relative MSE", fontsize=14)
        axis.set_ylabel("Permeability-field relative MSE", fontsize=14)
        axis.grid(which="both", linestyle="--", linewidth=0.5)
    figure.tight_layout()
    save_figure(figure, output_dir, "fig7_a" if case.testcase == 1 else "fig7_b")


def posterior_weights(
    observation_library: np.ndarray,
    observation: np.ndarray,
    var_v: float,
    var_y: float,
) -> np.ndarray:
    mean = observation_library.mean(axis=0)
    std = np.maximum(observation_library.std(axis=0), 1.0e-12)
    normalized_library = (observation_library - mean) / std
    normalized_observation = (observation - mean) / std
    observation_variance = (var_v + var_y) / std**2
    differences = normalized_library - normalized_observation[None, :]
    log_weights = -0.5 * np.sum(
        np.log(2.0 * np.pi * observation_variance)[None, :]
        + differences**2 / observation_variance[None, :],
        axis=1,
    )
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.sum(weights)


def gaussian_pdf_matrix(
    grid: np.ndarray,
    centers: np.ndarray,
    variance: float,
) -> np.ndarray:
    return np.exp(
        -0.5 * (grid[:, None] - centers[None, :]) ** 2 / variance
    ) / np.sqrt(2.0 * np.pi * variance)


def exact_gmm_1d(
    centers: np.ndarray,
    weights: np.ndarray,
    component_variance: float,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if grid is None:
        weighted_mean = float(np.dot(weights, centers))
        total_std = np.sqrt(
            np.dot(weights, (centers - weighted_mean) ** 2) + component_variance
        )
        padding = 0.15 * max(total_std, 1.0e-8)
        grid = np.linspace(centers.min() - padding, centers.max() + padding, 400)
    density = np.zeros_like(grid)
    for start in range(0, centers.size, 1024):
        stop = min(start + 1024, centers.size)
        density += gaussian_pdf_matrix(
            grid,
            centers[start:stop],
            component_variance,
        ) @ weights[start:stop]
    return grid, density


def density_mass_thresholds(
    density: np.ndarray,
    mass_levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> np.ndarray:
    sorted_density = np.sort(np.asarray(density).ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative /= cumulative[-1]
    thresholds = [
        sorted_density[
            min(int(np.searchsorted(cumulative, level)), sorted_density.size - 1)
        ]
        for level in mass_levels
    ]
    return np.unique(np.sort(thresholds))


def exact_gmm_2d(
    centers: np.ndarray,
    weights: np.ndarray,
    variance_x: float,
    variance_y_value: float,
    grid_size: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None] * centers, axis=0)
    centered = centers - mean
    covariance = (weights[:, None] * centered).T @ centered
    covariance += np.diag([variance_x, variance_y_value])
    padding = 0.15 * np.sqrt(np.maximum(np.diag(covariance), 1.0e-12))
    x_values = np.linspace(centers[:, 0].min() - padding[0], centers[:, 0].max() + padding[0], grid_size)
    y_values = np.linspace(centers[:, 1].min() - padding[1], centers[:, 1].max() + padding[1], grid_size)
    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    for start in range(0, centers.shape[0], 1024):
        stop = min(start + 1024, centers.shape[0])
        pdf_x = gaussian_pdf_matrix(x_values, centers[start:stop, 0], variance_x)
        pdf_y = gaussian_pdf_matrix(y_values, centers[start:stop, 1], variance_y_value)
        density += (pdf_y * weights[None, start:stop]) @ pdf_x.T
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, density, density_mass_thresholds(density)


def sample_kde_1d(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(samples)(grid)


def sample_kde_2d(
    samples: np.ndarray,
    grid_size: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(-2.0, 2.0, grid_size)
    y_values = np.linspace(-2.0, 2.0, grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = gaussian_kde(samples.T)(positions).reshape(grid_size, grid_size)
    return x_grid, y_grid, density, density_mass_thresholds(density)


def set_density_ylim(axis: plt.Axes) -> None:
    maximum = max(
        [float(np.nanmax(line.get_ydata())) for line in axis.lines] + [1.0e-6]
    )
    axis.set_ylim(0.0, 1.08 * maximum)


def plot_coefficient_comparison(
    coefficient_library: np.ndarray,
    weights: np.ndarray,
    nn_samples: np.ndarray,
    mcmc_samples: np.ndarray,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(6.2, 6.2))
    colors = {"Bayesian GMM": "blue", "NN": "red", "MCMC": "green"}
    density_grid = np.linspace(-2.0, 2.0, 400)

    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            if column > row:
                axis.axis("off")
                continue
            if row == column:
                _, exact_density = exact_gmm_1d(
                    coefficient_library[:, row],
                    weights,
                    variance_u,
                    density_grid,
                )
                axis.plot(density_grid, exact_density, "b--", linewidth=1.4)
                axis.plot(density_grid, sample_kde_1d(nn_samples[:, row], density_grid), color="red", linewidth=1.2)
                axis.plot(density_grid, sample_kde_1d(mcmc_samples[:, row], density_grid), color="green", linewidth=1.8)
                axis.set_xlim(-2.0, 2.0)
                axis.set_yticks([])
                set_density_ylim(axis)
            else:
                x_grid, y_grid, density, levels = exact_gmm_2d(
                    coefficient_library[:, [column, row]],
                    weights,
                    variance_u,
                    variance_u,
                )
                axis.contour(x_grid, y_grid, density, levels=levels, colors="blue", linestyles="--", linewidths=1.2)
                for samples, color, width in (
                    (nn_samples[:, [column, row]], "red", 1.0),
                    (mcmc_samples[:, [column, row]], "green", 0.8),
                ):
                    x_kde, y_kde, kde, kde_levels = sample_kde_2d(samples)
                    axis.contour(x_kde, y_kde, kde, levels=kde_levels, colors=color, linewidths=width)
                axis.set_xlim(-2.0, 2.0)
                axis.set_ylim(-2.0, 2.0)

            if row == 3:
                axis.set_xlabel(coefficient_labels[column], fontsize=12)
            else:
                axis.set_xticks([])
            if column == 0 and row != column:
                axis.set_ylabel(coefficient_labels[row], fontsize=12)
            elif row != column:
                axis.set_yticks([])
            axis.grid(False)

    handles = [
        Line2D([0], [0], color=color, linestyle="--" if label == "Bayesian GMM" else "-", label=label)
        for label, color in colors.items()
    ]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=16,
        handlelength=2.2,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "fig8")


def plot_prior_2d_marginals(
    prior_coefficients: np.ndarray,
    output_dir: Path,
) -> None:
    prior_coefficients = as_coefficients(
        prior_coefficients,
        "prior coefficients",
    )
    figure, axes = plt.subplots(4, 4, figsize=(8.5, 8.5))
    density_grid = np.linspace(-2.0, 2.0, 400)

    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            if column > row:
                axis.axis("off")
                continue

            if row == column:
                density = sample_kde_1d(
                    prior_coefficients[:, row],
                    density_grid,
                )
                axis.plot(density_grid, density, color="blue", linewidth=2.0)
                axis.set_xlim(-2.0, 2.0)
                axis.set_yticks([])
                set_density_ylim(axis)
            else:
                x_grid, y_grid, density, _ = sample_kde_2d(
                    prior_coefficients[:, [column, row]]
                )
                contour_levels = np.linspace(
                    0.08 * float(np.max(density)),
                    0.92 * float(np.max(density)),
                    8,
                )
                axis.contour(
                    x_grid,
                    y_grid,
                    density,
                    levels=contour_levels,
                    colors="blue",
                    linewidths=1.5,
                )
                axis.set_xlim(-2.0, 2.0)
                axis.set_ylim(-2.0, 2.0)

            axis.tick_params(labelsize=13)
            if row == 3:
                axis.set_xlabel(coefficient_labels[column], fontsize=18)
                axis.set_xticks([-2.0, 0.0, 2.0])
            else:
                axis.set_xticks([])
            if column == 0 and row != column:
                axis.set_ylabel(coefficient_labels[row], fontsize=18)
                axis.set_yticks([-2.0, 0.0, 2.0])
            elif row != column:
                axis.set_yticks([])
            axis.grid(False)

    prior_handle = Line2D(
        [0],
        [0],
        color="blue",
        linewidth=2.5,
        label="Prior",
    )
    figure.legend(
        [prior_handle],
        [prior_handle.get_label()],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        frameon=False,
        fontsize=16,
        handlelength=2.8,
    )
    figure.tight_layout(pad=0.8)
    save_figure(figure, output_dir, "prior_2D_marginal")


def plot_exact_ablation(
    coefficient_library: np.ndarray,
    observation_library: np.ndarray,
    observation: np.ndarray,
    sweep: str,
    testcase: int,
    output_dir: Path,
) -> None:
    if sweep == "u":
        values = (1.0e-1, 1.0e-2, 1.0e-3)
        results = [
            (
                value,
                posterior_weights(observation_library, observation, variance_v, variance_y),
                value,
            )
            for value in values
        ]
        stem = f"testcase{testcase}_ablations_varU"
        symbol = "U"
    else:
        values = (1.0e-4, 1.0e-5, 1.0e-6)
        results = [
            (
                value,
                posterior_weights(observation_library, observation, value, variance_y),
                variance_u,
            )
            for value in values
        ]
        stem = f"testcase{testcase}_ablations_varV"
        symbol = "V"

    colors = ("blue", "red", "green")
    styles = ("--", "--", "-")
    alphas = (0.65, 0.70, 0.65)
    draw_order = (2, 1, 0)
    figure, axes = plt.subplots(4, 4, figsize=(6.2, 6.2))

    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            if column > row:
                axis.axis("off")
                continue

            if row == column:
                for index in draw_order:
                    _, weights, component_variance = results[index]
                    grid, density = exact_gmm_1d(
                        coefficient_library[:, row],
                        weights,
                        component_variance,
                    )
                    axis.plot(
                        grid,
                        density,
                        color=colors[index],
                        linestyle=styles[index],
                        linewidth=1.5,
                        alpha=alphas[index],
                    )
                axis.set_xlim(-2.0, 2.0)
                axis.set_yticks([])
                set_density_ylim(axis)
            else:
                for index in draw_order:
                    _, weights, component_variance = results[index]
                    x_grid, y_grid, density, levels = exact_gmm_2d(
                        coefficient_library[:, [column, row]],
                        weights,
                        component_variance,
                        component_variance,
                    )
                    axis.contour(
                        x_grid,
                        y_grid,
                        density,
                        levels=levels,
                        colors=[colors[index]],
                        linestyles=styles[index],
                        linewidths=1.4,
                        alpha=alphas[index],
                    )
                axis.set_xlim(-2.0, 2.0)
                axis.set_ylim(-2.0, 2.0)

            if row == 3:
                axis.set_xlabel(coefficient_labels[column], fontsize=12)
            else:
                axis.set_xticks([])
            if column == 0 and row != column:
                axis.set_ylabel(coefficient_labels[row], fontsize=12)
            elif row != column:
                axis.set_yticks([])
            axis.grid(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[index],
            linestyle=styles[index],
            linewidth=1.5,
            alpha=alphas[index],
            label=rf"$\sigma_{symbol}^2={value:.0e}$",
        )
        for index, value in enumerate(values)
    ]
    figure.legend(handles, [handle.get_label() for handle in handles], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=15)
    figure.tight_layout()
    save_figure(figure, output_dir, stem)


def generate_all_plots(
    input_dir: Path,
    output_dir: Path,
    solve_missing: bool = False,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    solved_dir = input_dir / "solved_plot_samples"
    solved_dir.mkdir(parents=True, exist_ok=True)
    data_dir = input_dir / "data"

    sensor_points = np.asarray(
        load_array(data_dir / "10_sampling_locs_x_y_grid1.npy", "sensor points"),
        dtype=np.float64,
    )
    if sensor_points.shape != (sensor_count, 2):
        raise ValueError(f"Sensor points must have shape (10, 2); got {sensor_points.shape}")

    coefficient_library = as_coefficients(
        load_array(data_dir / "b_mn_samples_20000.npy", "coefficient library"),
        "coefficient library",
    )
    observation_library = as_observations(
        load_array(
            data_dir / "sampled_solutions_20000_10_locs.npy",
            "observation library",
        ),
        "observation library",
    )
    if coefficient_library.shape[0] != observation_library.shape[0]:
        raise ValueError("Coefficient and observation libraries must have equal rows")

    prior_count = min(sample_limit, coefficient_library.shape[0])
    prior_coefficients = coefficient_library[:prior_count]
    prior_sensors = observation_library[:prior_count]
    plot_prior_2d_marginals(prior_coefficients, output_dir)
    x_grid, y_grid, basis = coefficient_basis()

    cases = {
        testcase: prepare_case(
            testcase,
            input_dir,
            solved_dir,
            prior_coefficients,
            prior_sensors,
            sensor_points,
            basis,
            solve_missing,
        )
        for testcase in (1, 2)
    }

    for case in cases.values():
        plot_fields(case, x_grid, y_grid, basis, sensor_points, output_dir)
    plot_combined_sensor_errors(cases, output_dir)
    for case in cases.values():
        plot_relative_mse(case, basis, output_dir)

    rng = np.random.default_rng(1234)
    noisy_observations = observation_library + rng.normal(
        0.0,
        observation_noise_std,
        size=observation_library.shape,
    )
    testcase_one_weights = posterior_weights(
        noisy_observations,
        cases[1].true_sensors,
        variance_v,
        variance_y,
    )
    plot_coefficient_comparison(
        coefficient_library,
        testcase_one_weights,
        cases[1].nn_coefficients,
        cases[1].mcmc_coefficients,
        output_dir,
    )

    for testcase in (1, 2):
        for sweep in ("u", "v"):
            plot_exact_ablation(
                coefficient_library,
                noisy_observations,
                cases[testcase].true_sensors,
                sweep,
                testcase,
                output_dir,
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    generate_all_plots(args.input_dir, output_dir, args.solve_missing)


if __name__ == "__main__":
    main()
