"""Paired four-configuration ORBITA ablation benchmark.

Writes only data/benchmarks/ablation and figures/ablation.  Every configuration
uses the same deterministic cases and the same Cowell state for each row.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import torch

from benchmark import eci_to_ric_error, get_model_instance
from generate_base_dataset import MIN_SAFE_PERIGEE
from ml.dataset import OrbitalDataset
from physics.analytical import compute_general_solution
from physics.kepler import get_keplerian
from physics.oracle import (
    J2,
    J3,
    MU,
    R_EQ,
    coe_to_eci,
    coe_to_mee,
    eci_to_coe,
    get_ground_truth,
    mee_to_coe,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_RESNET = ROOT / "data" / "benchmarks" / "ablation_resnet"
OUT = OUT_RESNET
FIG = ROOT / "figures" / "ablation_resnet"
NAMES = {
    "A": "ESTHER",
    "B": "ESTHER + ResNet global",
    "C": "ESTHER + expertos ResNet base",
    "D": "ESTHER + expertos ResNet ajustados",
}
COMPONENTS = [
    ("Radial_m", "Radial"),
    ("InTrack_m", "In-track"),
    ("CrossTrack_m", "Cross-track"),
]


def baseline_step(coe, dt):
    sma, ecc, inc, raan, aop, ta = coe
    r0, v0 = coe_to_eci(MU, *coe)
    delta_r, delta_v = compute_general_solution(
        J2,
        J3,
        R_EQ,
        sma,
        ecc,
        inc,
        raan,
        aop,
        ta,
        np.zeros(6),
        np.sqrt(MU / sma**3),
        dt,
    )
    r_kep, v_kep = get_keplerian(MU, r0, v0, dt)
    return r_kep + delta_r, v_kep + delta_v


def corrected_step(coe, dt, model, dataset):
    r_esther, v_esther = baseline_step(coe, dt)
    p, f, g, h, k, ell = coe_to_mee(*coe)
    raw = np.array([p, f, g, h, k, np.sin(ell), np.cos(ell), dt])
    x = torch.tensor((raw - dataset.x_mean) / dataset.x_std, dtype=torch.float32)[
        None, :
    ]
    with torch.no_grad():
        residual = model(x).numpy()[0] * dataset.y_std + dataset.y_mean
    corrected = coe_to_mee(*eci_to_coe(MU, r_esther, v_esther)) + residual
    return coe_to_eci(MU, *mee_to_coe(*corrected))


def load_model(path, dataset_path):
    model = get_model_instance("resnet")
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model, OrbitalDataset(dataset_path)


def region_for(coe):
    altitude = (coe[0] - R_EQ) / 1000.0
    ecc = coe[1]
    inc = np.degrees(coe[2])
    for path in sorted(
        (ROOT / "models" / "resnet").glob("**/*.pth")
    ):
        if path.name.endswith("_finetuned.pth") or "300-2000_0.0000-0.1000_0-90" in path.name:
            continue
        tokens = path.stem.replace("orbita_predictor_resnet_", "").split("_")
        if len(tokens) != 3:
            continue
        a, e, i = (tuple(map(float, token.split("-"))) for token in tokens)
        if a[0] <= altitude <= a[1] and e[0] <= ecc <= e[1] and i[0] <= inc <= i[1]:
            return "_".join(tokens)
    raise ValueError(f"No valid expert region for {altitude=}, {ecc=}, {inc=}")


def resources():
    global_model_path = (
        ROOT
        / "models"
        / "resnet"
        / "base"
        / "orbita_predictor_resnet_300-2000_0.0000-0.1000_0-90.pth"
    )
    if not global_model_path.exists():
        global_model_path = (
            ROOT
            / "models"
            / "resnet"
            / "orbita_predictor_resnet_300-2000_0.0000-0.1000_0-90.pth"
        )

    global_data_path = (
        ROOT
        / "data"
        / "datasets"
        / "training"
        / "orbita_dataset_300-2000_0.0000-0.1000_0-90.csv"
    )
    if not global_data_path.exists():
        global_data_path = (
            ROOT / "data" / "orbita_dataset_300-2000_0.0000-0.1000_0-90.csv"
        )

    global_model, global_data = load_model(global_model_path, global_data_path)

    experts = {}
    for path in (ROOT / "models" / "resnet").glob("**/*.pth"):
        if path.name.endswith("_finetuned.pth"):
            continue
        tokens = path.stem.replace("orbita_predictor_resnet_", "").split("_")
        if len(tokens) != 3:
            continue
        region = "_".join(tokens)
        if region == "300-2000_0.0000-0.1000_0-90":
            continue

        data = ROOT / "data" / "datasets" / "training" / f"orbita_dataset_{region}.csv"
        if not data.exists():
            data = ROOT / "data" / f"orbita_dataset_{region}.csv"

        finetuned = (
            ROOT / "models" / "resnet" / "finetuned" / f"{path.stem}_finetuned.pth"
        )
        if not finetuned.exists():
            finetuned = path.with_name(path.stem + "_finetuned.pth")

        if data.exists() and finetuned.exists():
            experts[region] = (*load_model(path, data), *load_model(finetuned, data))

    if len(experts) != 38:
        raise RuntimeError(f"Expected 38 expert regions, found {len(experts)}")
    return global_model, global_data, experts


def make_cases(count, seed):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < count:
        sma = rng.uniform(R_EQ + 300e3, R_EQ + 2000e3)
        ecc = rng.uniform(0.0, 0.1)
        if sma * (1.0 - ecc) < MIN_SAFE_PERIGEE:
            continue
        cases.append(
            (
                sma,
                ecc,
                rng.uniform(0, np.pi / 2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
            )
        )
    return cases


def write_cases(path, cases):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Case_ID", "SMA_m", "ECC", "INC_rad", "RAAN_rad", "AOP_rad", "TA_rad"]
        )
        for index, coe in enumerate(cases, 1):
            writer.writerow([f"case_{index:05d}", *coe])


def row(case_id, time_s, coe, truth, estimates, region):
    r_true, v_true = truth
    values = [case_id, time_s, *coe, region]
    for key in "ABCD":
        err = eci_to_ric_error(r_true, v_true, estimates[key][0])
        values.extend(err)
    return values


def run_time(cases, global_model, global_data, experts):
    path = OUT / "temporal_paired.csv"
    header = [
        "Case_ID",
        "Time_s",
        "SMA0_m",
        "ECC0",
        "INC0_rad",
        "RAAN0_rad",
        "AOP0_rad",
        "TA0_rad",
        "Region",
    ] + [f"{key}_{col}" for key in "ABCD" for col, _ in COMPONENTS]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, initial in enumerate(cases, 1):
            states = {key: np.array(initial) for key in "ABCD"}
            region = region_for(initial)
            base_model, base_ds, fine_model, fine_ds = experts[region]
            for step in range(1, 17):
                time_s = step * 900
                truth = get_ground_truth(*initial, time_s)
                estimates = {}
                estimates["A"] = baseline_step(states["A"], 900)
                estimates["B"] = corrected_step(
                    states["B"], 900, global_model, global_data
                )
                estimates["C"] = corrected_step(states["C"], 900, base_model, base_ds)
                estimates["D"] = corrected_step(states["D"], 900, fine_model, fine_ds)
                writer.writerow(
                    row(f"case_{index:05d}", time_s, initial, truth, estimates, region)
                )
                for key, state in estimates.items():
                    states[key] = eci_to_coe(MU, *state)
            if index % 100 == 0:
                print(f"temporal {index}/{len(cases)}", flush=True)


def run_space(cases, global_model, global_data, experts):
    path = OUT / "espacial_paired.csv"
    header = [
        "Case_ID",
        "Time_s",
        "SMA0_m",
        "ECC0",
        "INC0_rad",
        "RAAN0_rad",
        "AOP0_rad",
        "TA0_rad",
        "Region",
    ] + [f"{key}_{col}" for key in "ABCD" for col, _ in COMPONENTS]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, initial in enumerate(cases, 1):
            region = region_for(initial)
            base_model, base_ds, fine_model, fine_ds = experts[region]
            truth = get_ground_truth(*initial, 900)
            estimates = {
                "A": baseline_step(initial, 900),
                "B": corrected_step(initial, 900, global_model, global_data),
                "C": corrected_step(initial, 900, base_model, base_ds),
                "D": corrected_step(initial, 900, fine_model, fine_ds),
            }
            writer.writerow(
                row(f"case_{index:05d}", 900, initial, truth, estimates, region)
            )
            if index % 1000 == 0:
                print(f"espacial {index}/{len(cases)}", flush=True)


def load_rows(path):
    path = Path(path)
    if not path.exists():
        if "temporal" in path.name:
            alt = path.with_name("time_paired.csv")
            if alt.exists():
                path = alt
        elif "espacial" in path.name:
            alts = [
                path.with_name("space_paired.csv"),
                path.with_name("spatial_paired.csv"),
            ]
            for a in alts:
                if a.exists():
                    path = a
                    break
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )


def metric_summary(rows, temporal=False):
    output = {}
    times = sorted(set(rows["Time_s"])) if temporal else [900]
    for key in "ABCD":
        output[key] = {}
        for time_s in times:
            subset = rows[rows["Time_s"] == time_s]
            output[key][str(int(time_s))] = {
                component: float(np.mean(np.abs(subset[f"{key}_{column}"])))
                for column, component in COMPONENTS
            }
            if not temporal:
                output[key][str(int(time_s))].update(
                    {
                        f"P95_{component}": float(
                            np.percentile(np.abs(subset[f"{key}_{column}"]), 95)
                        )
                        for column, component in COMPONENTS
                    }
                )
    return output


def improvements(summary, time_s, p95=False):
    labels = [
        ("A_B", "A", "B"),
        ("B_C", "B", "C"),
        ("C_D", "C", "D"),
        ("A_D", "A", "D"),
    ]
    result = {}
    for label, before, after in labels:
        result[label] = {}
        for _, component in COMPONENTS:
            metric = f"P95_{component}" if p95 else component
            x, y = (
                summary[before][str(time_s)][metric],
                summary[after][str(time_s)][metric],
            )
            result[label][component] = 100 * (x - y) / x
    return result


def region_summary(rows):
    records = []
    for region in sorted(set(rows["Region"])):
        subset = rows[rows["Region"] == region]
        record = {"Region": region, "N": int(len(subset))}
        for key in "BCD":
            for column, component in COMPONENTS:
                values = np.abs(subset[f"{key}_{column}"])
                record[f"{key}_Mean_{component}"] = float(np.mean(values))
                record[f"{key}_P95_{component}"] = float(np.percentile(values, 95))
        for before, after in [("B", "C"), ("C", "D"), ("B", "D")]:
            for _, component in COMPONENTS:
                record[f"{before}_{after}_{component}_improvement_pct"] = (
                    100
                    * (
                        record[f"{before}_Mean_{component}"]
                        - record[f"{after}_Mean_{component}"]
                    )
                    / record[f"{before}_Mean_{component}"]
                )
        records.append(record)
    return records


def write_region(records):
    with (OUT / "regiones.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


class InteractiveAblationZoomManager:
    """Interactive Zoom Manager for ResNet ablation figure.

    Allows left-click drag to select ROI region and right-click drag to
    manually place/size the inset zoom sub-axes window on any of the 3
    subplots.
    """

    def __init__(self, fig, axes, data_dict, colors, names):
        self.fig = fig
        self.axes = axes  # 3 subplots: Radial, Tangencial, Normal
        self.data_dict = data_dict
        self.colors = colors
        self.names = names

        # Initial state per subplot index (0: Radial, 1: Tangencial, 2: Normal)
        # Default presets for immediate visual display:
        self.state = {
            0: {
                "roi": [0.5, 1.5, 80.0, 400.0],
                "inset_pos": [0.55, 0.12, 0.40, 0.45],
                "axins": None,
                "indicator": None,
            },
            1: {
                "roi": [0.5, 1.5, 100.0, 2000.0],
                "inset_pos": [0.55, 0.12, 0.40, 0.45],
                "axins": None,
                "indicator": None,
            },
            2: {
                "roi": [0.5, 1.5, 1.2, 5.0],
                "inset_pos": [0.55, 0.12, 0.40, 0.45],
                "axins": None,
                "indicator": None,
            },
        }

        self.roi_selectors = []
        self.pos_selectors = []
        self._setup_selectors()

        # Build initial insets for all 3 subplots
        for idx in range(3):
            self.update_inset(idx)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _setup_selectors(self):
        for i, ax in enumerate(self.axes):
            # Left click -> ROI area selector
            roi_sel = RectangleSelector(
                ax,
                onselect=self._make_on_roi_select(i),
                useblit=True,
                button=[1],  # Left click
                interactive=True,
                props=dict(facecolor="#2A9D8F", alpha=0.2, edgecolor="#2A9D8F"),
            )
            self.roi_selectors.append(roi_sel)

            # Right click -> Inset position/size selector
            pos_sel = RectangleSelector(
                ax,
                onselect=self._make_on_pos_select(i),
                useblit=True,
                button=[3],  # Right click
                interactive=True,
                props=dict(facecolor="#B13A3A", alpha=0.2, edgecolor="#B13A3A"),
            )
            self.pos_selectors.append(pos_sel)

    def _make_on_roi_select(self, idx):
        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if x1 is None or x2 is None or y1 is None or y2 is None:
                return
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            if xmin == xmax or ymin == ymax:
                return
            self.state[idx]["roi"] = [xmin, xmax, ymin, ymax]
            self.update_inset(idx)

        return on_select

    def _make_on_pos_select(self, idx):
        def on_select(eclick, erelease):
            ax = self.axes[idx]
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if x1 is None or x2 is None or y1 is None or y2 is None:
                return

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            rx1 = (x1 - xlim[0]) / (xlim[1] - xlim[0])
            rx2 = (x2 - xlim[0]) / (xlim[1] - xlim[0])

            log_y1, log_y2 = np.log10(y1), np.log10(y2)
            log_ylim = np.log10(ylim)
            ry1 = (log_y1 - log_ylim[0]) / (log_ylim[1] - log_ylim[0])
            ry2 = (log_y2 - log_ylim[0]) / (log_ylim[1] - log_ylim[0])

            pos_x = max(0.02, min(rx1, rx2))
            pos_y = max(0.02, min(ry1, ry2))
            pos_w = max(0.1, abs(rx2 - rx1))
            pos_h = max(0.1, abs(ry2 - ry1))

            self.state[idx]["inset_pos"] = [pos_x, pos_y, pos_w, pos_h]
            self.update_inset(idx)

        return on_select

    def update_inset(self, idx):
        ax = self.axes[idx]
        st = self.state[idx]
        if st["roi"] is None:
            return

        xmin, xmax, ymin, ymax = st["roi"]
        pos = st["inset_pos"]

        # Clear existing inset & indicators
        if st["indicator"] is not None:
            try:
                st["indicator"].remove()
            except Exception:
                pass
            st["indicator"] = None

        if st["axins"] is not None:
            try:
                st["axins"].remove()
            except Exception:
                pass
            st["axins"] = None

        # Create inset axes
        axins = ax.inset_axes(pos)
        st["axins"] = axins

        for key in "ABCD":
            th, vals = self.data_dict[idx][key]
            axins.plot(th, vals, color=self.colors[key], linewidth=1.5)

        axins.set_xlim(xmin, xmax)
        axins.set_ylim(ymin, ymax)
        axins.set_yscale("log")
        axins.grid(True, which="both", alpha=0.3)
        axins.tick_params(axis="both", labelsize=7)

        # Indicate zoom connecting region & lines
        try:
            st["indicator"] = ax.indicate_inset_zoom(axins, edgecolor="#444444", alpha=0.7)
        except Exception:
            pass

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key in ["s", "S"]:
            print(" [saved] Saving ablation figures with insets...")
            self.fig.savefig(FIG / "resnet_ablation_time_domain.svg")
            self.fig.savefig(FIG / "resnet_ablation_time_domain.png", dpi=600)
            print(" [complete] Saved successfully to figures/resnet_ablation_time_domain.png!")


def figures(time_rows, space_summary, regions, interactive=False):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    colors = {
        "A": "#B13A3A",  # ESTHER
        "B": "#3569A8",  # ResNet global
        "C": "#8F12A6",  # Expertos ResNet base
        "D": "#2A9D8F",  # Expertos ResNet ajustados
    }
    components_plot = [
        ("Radial_m", "Error radial medio [m]"),
        ("InTrack_m", "Error tangencial medio [m]"),
        ("CrossTrack_m", "Error normal medio [m]"),
    ]
    time_hours = np.array(sorted(set(time_rows["Time_s"]))) / 3600

    data_dict = {}

    for idx, (axis, (column, ylabel)) in enumerate(zip(axes, components_plot)):
        data_dict[idx] = {}
        for key in "ABCD":
            values = [
                np.mean(np.abs(time_rows[time_rows["Time_s"] == t][f"{key}_{column}"]))
                for t in sorted(set(time_rows["Time_s"]))
            ]
            data_dict[idx][key] = (time_hours, values)
            axis.plot(
                time_hours,
                values,
                label=NAMES[key],
                color=colors[key],
            )
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_title(
        "Estudio de ablación de ResNet: acumulación de error secular temporal",
        pad=48,
        fontweight="bold",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        fontsize=7.8,
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    axes[-1].set_xlabel("Tiempo [h]")
    axes[-1].set_xlim(left=time_hours.min(), right=time_hours.max())
    fig.tight_layout()

    if interactive:
        # Attach interactive zoom manager
        InteractiveAblationZoomManager(
            fig, axes, data_dict, colors, NAMES
        )
        print("\n" + "=" * 80)
        print(" INTERACTIVE ZOOM MODULE FOR RESNET ABLATION")
        print("  - Left Click + Drag: Select ROI area to zoom")
        print("  - Right Click + Drag: Position inset zoom sub-axes")
        print("  - Press 's' key: Save figure and continue")
        print("=" * 80 + "\n")
        plt.show()

    fig.savefig(FIG / "resnet_ablation_time_domain.svg")
    fig.savefig(FIG / "resnet_ablation_time_domain.png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-cases", type=int, default=10000)
    parser.add_argument("--space-cases", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--skip-time", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument(
        "--only-figures",
        action="store_true",
        help="Load existing data and instantly generate figures",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive window to adjust zoom insets in subplots",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Fast path: If only generating figures or skipping computations
    if args.only_figures or (args.skip_time and args.skip_space):
        time_rows = load_rows(OUT / "temporal_paired.csv")
        figures(time_rows, {}, [], interactive=args.interactive)
        return

    global_model, global_data, experts = resources()
    time_cases = make_cases(args.time_cases, args.seed)
    space_cases = make_cases(args.space_cases, args.seed + 1)
    if not (OUT / "casos_temporales.csv").exists():
        write_cases(OUT / "casos_temporales.csv", time_cases)
    if not (OUT / "casos_espaciales.csv").exists():
        write_cases(OUT / "casos_espaciales.csv", space_cases)

    if not args.skip_time:
        run_time(time_cases, global_model, global_data, experts)
    if not args.skip_space:
        run_space(space_cases, global_model, global_data, experts)
    time_rows = load_rows(OUT / "temporal_paired.csv")
    space_rows = load_rows(OUT / "espacial_paired.csv")
    temporal = metric_summary(time_rows, temporal=True)
    spatial = metric_summary(space_rows)
    regions = region_summary(space_rows)
    write_region(regions)
    figures(time_rows, spatial, regions, interactive=args.interactive)
    report = {
        "configurations": NAMES,
        "protocol": {
            "same_cases_and_oracle": True,
            "seed": args.seed,
            "temporal_cases": args.time_cases,
            "space_cases": args.space_cases,
        },
        "temporal_mean_abs": temporal,
        "spatial": spatial,
        "temporal_final_improvements_pct": improvements(temporal, 14400),
        "spatial_mean_improvements_pct": improvements(spatial, 900),
        "spatial_p95_improvements_pct": improvements(spatial, 900, p95=True),
        "regions": {
            "count": len(regions),
            "B_to_C_improved_pct": 100
            * np.mean([r["B_C_Radial_improvement_pct"] > 0 for r in regions]),
            "C_to_D_improved_pct": 100
            * np.mean([r["C_D_Radial_improvement_pct"] > 0 for r in regions]),
            "B_to_D_improved_pct": 100
            * np.mean([r["B_D_Radial_improvement_pct"] > 0 for r in regions]),
        },
    }
    with (OUT / "resumen.json").open("w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
