#!/usr/bin/env python3
"""Plot the saved C1--C9 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def write_latex_table(rows: list[dict[str, float | int | str]], path: Path) -> None:
    lines = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{>{\small}c|ccc|ccc}",
        r"\hline",
        (
            r" & $K$ & $\sigma^2_{\bU}$ & $\sigma^2_{\bY}$ "
            r"& $e_{\rm exact}$ & $e_{\rm GMM}$ & $e_{\rm BGMM}$\\"
        ),
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['K']} & "
            f"{row['sigma_u_squared']:g} & "
            f"{row['sigma_y_squared']:g} & "
            f"{row['epsilon_exact']:.2e} & "
            f"{row['epsilon_gmm']:.2e} & "
            f"{row['epsilon_bgmm']:.2e} \\\\"
        )
        if row["label"] in ("C3", "C6", "C9"):
            lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{center}"])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("bimodal_c1_c9_results"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = [args.input_dir / f"C{index}.npz" for index in range(1, 10)]
    missing = [path for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing C1--C9 result files:\n" + "\n".join(map(str, missing))
        )

    table_rows: list[dict[str, float | int | str]] = []
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    for index, (axis, result_path) in enumerate(zip(axes.flat, result_paths)):
        with np.load(result_path, allow_pickle=False) as result:
            grid = result["grid"]
            generated = result["generated_samples"]
            label = index == 0
            best_index = int(np.argmin(result["epsilon_bgmm"]))
            table_rows.append(
                {
                    "label": str(result["config_label"].item()),
                    "K": int(result["K"].item()),
                    "sigma_u_squared": float(result["sigma_u_squared"].item()),
                    "sigma_y_squared": float(result["sigma_y_squared"].item()),
                    "epsilon_exact": float(result["epsilon_exact"][best_index]),
                    "epsilon_gmm": float(result["epsilon_gmm"][best_index]),
                    "epsilon_bgmm": float(result["epsilon_bgmm"][best_index]),
                }
            )

            axis.plot(
                grid,
                result["bgmm_density"],
                "b-",
                linewidth=2,
                alpha=0.9,
                label="Bayesian GMM: $p_{BGMM}$" if label else None,
            )
            axis.plot(
                grid,
                result["gmm_density"],
                "g--",
                linewidth=2,
                alpha=0.9,
                label="GMM conditional: $p_{GMM}$" if label else None,
            )
            axis.plot(
                grid,
                result["exact_density"],
                "r:",
                linewidth=2,
                alpha=0.9,
                label="True posterior: $p_{exact}$" if label else None,
            )
            axis.hist(
                generated[:, 0],
                bins=50,
                density=True,
                alpha=0.4,
                color="blue",
                histtype="stepfilled",
                label="Diffusion model" if label else None,
            )
            axis.set_title(
                rf"C{index + 1}: $K={int(result['K'].item())}$, "
                rf"$\sigma_U^2={float(result['sigma_u_squared'].item()):g}$, "
                rf"$\sigma_Y^2={float(result['sigma_y_squared'].item()):g}$",
                fontsize=14,
            )

        axis.set_xlim(-2, 2)
        axis.set_ylim(0, 1.8)
        axis.grid(which="both", linestyle="--", linewidth=0.5)
        axis.set_xlabel("$U$", fontsize=13)
        axis.set_ylabel("Density", fontsize=13)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        fontsize=14,
    )
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.11)

    png_path = output_dir / "fig1_bimodal.png"
    pdf_path = output_dir / "fig1_bimodal.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    table_path = args.input_dir / "summary_table.tex"
    write_latex_table(table_rows, table_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
