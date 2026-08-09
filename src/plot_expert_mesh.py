"""Create expert mesh validity grid figure and regional improvement maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "ablation_resnet"
DATA_RESNET = ROOT / "data/benchmarks/ablation_resnet/regions.csv"
DATA_OLD = ROOT / "data/benchmarks/ablation_resnet/regiones.csv"
DATA_LEGACY = ROOT / "data/benchmarks/ablation/regiones.csv"
DATA = DATA_RESNET if DATA_RESNET.exists() else (DATA_OLD if DATA_OLD.exists() else DATA_LEGACY)

ALTITUDE_LABELS = [
    "300–640",
    "640–980",
    "980–1320",
    "1320–1660",
    "1660–2000",
]

ECCENTRICITY_LABELS = [
    "0–0.025",
    "0.025–0.050",
    "0.050–0.075",
    "0.075–0.100",
]

INCLINATION_LABELS = [
    r"$0^\circ \leq i < 45^\circ$",
    r"$45^\circ \leq i \leq 90^\circ$",
]

# The cell (altitude index 0, eccentricity index 3) is invalid
# in both inclination intervals.
INVALID_CELLS = {(0, 3)}

VALID_COLOR = "#2E7D32"
INVALID_COLOR = "#C62828"
GRID_COLOR = "0.78"


def plot_expert_mesh():
    """
    Generates the expert mesh discrimination figure marking valid cells (perigee requirement met)
    with green checkmarks and invalid cells with red crosses.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.8),
        sharey=True,
    )

    figure.suptitle(
        "Malla de la mezcla de expertos",
        fontsize=12,
        fontweight="bold",
    )

    for axis_index, axis in enumerate(axes):
        axis.set_title(INCLINATION_LABELS[axis_index], pad=10)

        # Draw the cell boundaries.
        for x_position in range(len(ECCENTRICITY_LABELS) + 1):
            axis.axvline(
                x_position - 0.5,
                color=GRID_COLOR,
                linewidth=0.8,
                zorder=0,
            )

        for y_position in range(len(ALTITUDE_LABELS) + 1):
            axis.axhline(
                y_position - 0.5,
                color=GRID_COLOR,
                linewidth=0.8,
                zorder=0,
            )

        # Mark every expert.
        for altitude_index in range(len(ALTITUDE_LABELS)):
            for eccentricity_index in range(len(ECCENTRICITY_LABELS)):
                is_valid = (
                    altitude_index,
                    eccentricity_index,
                ) not in INVALID_CELLS

                symbol = "✓" if is_valid else "×"
                color = VALID_COLOR if is_valid else INVALID_COLOR

                axis.text(
                    eccentricity_index,
                    altitude_index,
                    symbol,
                    color=color,
                    fontsize=19,
                    fontweight="bold",
                    fontfamily="DejaVu Sans",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        axis.set_xlim(-0.5, len(ECCENTRICITY_LABELS) - 0.5)
        axis.set_ylim(-0.5, len(ALTITUDE_LABELS) - 0.5)

        axis.set_xticks(range(len(ECCENTRICITY_LABELS)))
        axis.set_xticklabels(ECCENTRICITY_LABELS)

        axis.set_yticks(range(len(ALTITUDE_LABELS)))
        axis.set_yticklabels(ALTITUDE_LABELS)

        axis.set_xlabel("Excentricidad", labelpad=10)

        # Keep a simple frame consistent with the TFG figures.
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.45")

        axis.tick_params(
            direction="out",
            length=3,
            width=0.8,
            color="0.4",
        )

    axes[0].set_ylabel("Altitud nominal, $H$ [km]")

    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.82,
        bottom=0.23,
        wspace=0.12,
    )

    out_root = ROOT / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_root / "orbita_expert_mesh.svg", bbox_inches="tight")
    figure.savefig(out_root / "orbita_expert_mesh.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(" [saved] orbita_expert_mesh (.svg & .png)")


def matrix(frame, inclination, column, ALT, INC, ECC):
    result = np.full((len(ECC), len(ALT)), np.nan)
    for col, altitude in enumerate(ALT):
        for row, eccentricity in enumerate(ECC):
            region = f"{altitude}_{eccentricity}_{inclination}"
            selected = frame.loc[frame.Region == region, column]
            if not selected.empty:
                result[row, col] = selected.iloc[0]
    return result


def plot_regional_improvements():
    """Generates regional improvement maps if regions data exists."""
    if not DATA.exists():
        return
    df = pd.read_csv(DATA)
    columns = [
        ("B_C_Radial_improvement_pct", "B $\\rightarrow$ C"),
        ("B_D_Radial_improvement_pct", "B $\\rightarrow$ D"),
    ]
    ALT = ["300-640", "640-980", "980-1320", "1320-1660", "1660-2000"]
    INC = ["0-45", "45-90"]
    ECC = ["0.0000-0.0250", "0.0250-0.0500", "0.0500-0.0750", "0.0750-0.1000"]

    values = df[[item[0] for item in columns]].to_numpy()
    bound = np.ceil(np.nanmax(np.abs(values)) / 10) * 10
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    for row, (column, label) in enumerate(columns):
        for col, inclination in enumerate(INC):
            image = axes[row, col].imshow(
                matrix(df, inclination, column, ALT, INC, ECC),
                cmap=cmap,
                vmin=-bound,
                vmax=bound,
                origin="lower",
                aspect="auto",
            )
            axes[row, col].set_title(f"{label}; inclinación {inclination}°")
            alt_labels = [
                "300--640",
                "640--980",
                "980--1320",
                "1320--1660",
                "1660--2000",
            ]
            ecc_labels = [
                "0--0,025",
                "0,025--0,050",
                "0,050--0,075",
                "0,075--0,100",
            ]
            axes[row, col].set_xticks(range(len(ALT)), alt_labels, rotation=25)
            axes[row, col].set_yticks(range(len(ECC)), ecc_labels)
            if col == 0:
                axes[row, col].set_ylabel("Excentricidad [-]")
    fig.supxlabel("Altitud [km]")
    fig.tight_layout()
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.04, shrink=0.85)
    cbar.set_label("Mejora radial media [%]", labelpad=15)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "resnet_ablation_regional_improvement.svg", bbox_inches="tight")
    fig.savefig(OUT / "resnet_ablation_regional_improvement.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_expert_mesh()
    plot_regional_improvements()


if __name__ == "__main__":
    main()
