from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DOMAIN_ORDER = ["healthcare", "finance", "hr"]
DOMAIN_LABELS = {"healthcare": "Healthcare", "finance": "Finance", "hr": "HR"}
DOMAIN_COLORS = {"healthcare": "#1f78b4", "finance": "#33a02c", "hr": "#e31a1c"}
DOMAIN_MARKERS = {"healthcare": "o", "finance": "s", "hr": "^"}

MODEL_ORDER = ["gpt-5.4", "ministral-3-cloud"]
MODEL_LABELS = {"gpt-5.4": "GPT-5.4", "ministral-3-cloud": "Ministral-3-Cloud"}

CONFIG_ORDER = ["none", "pre", "runtime", "posthoc", "full"]
CONFIG_LABELS = {
    "none": "None",
    "pre": "Pre",
    "runtime": "Runtime",
    "posthoc": "Post-hoc",
    "full": "Full",
}
CONFIG_SHORT = {"none": "N", "pre": "P", "runtime": "R", "posthoc": "H", "full": "F"}
CONFIG_COLORS = {
    "none": "#4c566a",
    "pre": "#2a9d8f",
    "runtime": "#457b9d",
    "posthoc": "#f4a261",
    "full": "#d62828",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_full_vs_none_delta_plot(significance_df: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True, constrained_layout=True)
    handles = [
        Patch(facecolor=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
        for domain in DOMAIN_ORDER
    ]
    handles.append(Patch(facecolor="white", edgecolor="#444444", hatch="//", label="Underpowered"))

    min_delta = float(significance_df["R_delta_full_minus_none"].min())
    max_delta = float(significance_df["R_delta_full_minus_none"].max())
    x_min = min(-0.01, min_delta - 0.02)
    x_max = max_delta + 0.08

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = significance_df[significance_df["model"] == model].copy()
        subset["domain"] = pd.Categorical(subset["domain"], DOMAIN_ORDER, ordered=True)
        subset = subset.sort_values("domain")
        y_positions = list(range(len(subset)))[::-1]
        bars = ax.barh(
            y_positions,
            subset["R_delta_full_minus_none"],
            color=[DOMAIN_COLORS[d] for d in subset["domain"]],
            edgecolor="#333333",
            linewidth=0.8,
        )
        for bar, row in zip(bars, subset.itertuples(index=False), strict=True):
            if bool(row.underpowered):
                bar.set_hatch("//")
                bar.set_facecolor("#f7f7f7")
            note = "underpowered" if bool(row.underpowered) else f"p={row.p_value_permutation:.3f}"
            ax.text(
                float(row.R_delta_full_minus_none) + 0.004,
                bar.get_y() + bar.get_height() / 2,
                note,
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
            )
            ax.text(
                max(x_min + 0.005, float(row.R_delta_full_minus_none) * 0.45),
                bar.get_y() + bar.get_height() / 2,
                f"{float(row.R_delta_full_minus_none):+.3f}",
                va="center",
                ha="center",
                fontsize=8,
                color="white" if float(row.R_delta_full_minus_none) > 0.05 else "#111111",
                fontweight="bold",
            )

        ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"$R_{\mathrm{audited,certified}}$ delta (Full - None)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([DOMAIN_LABELS[d] for d in subset["domain"]], fontsize=10)

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)
    return _save_figure(fig, output_path)


def _make_latency_operating_points_plot(frontier_df: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True, constrained_layout=True)
    frontier_df = frontier_df.copy()
    frontier_df["latency_s"] = frontier_df["latency_avg_ms"] / 1000.0
    frontier_df["config"] = pd.Categorical(frontier_df["config"], CONFIG_ORDER, ordered=True)

    x_max = float(frontier_df["latency_s"].max()) * 1.08
    y_min = max(0.0, float(frontier_df["R_mean"].min()) - 0.03)
    y_max = min(1.0, float(frontier_df["R_mean"].max()) + 0.04)

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = frontier_df[frontier_df["model"] == model].copy()
        for domain in DOMAIN_ORDER:
            domain_subset = subset[subset["domain"] == domain].sort_values("config")
            if domain_subset.empty:
                continue
            ax.plot(
                domain_subset["latency_s"],
                domain_subset["R_mean"],
                color=DOMAIN_COLORS[domain],
                linewidth=1.0,
                alpha=0.25,
            )
            for row in domain_subset.itertuples(index=False):
                ax.scatter(
                    float(row.latency_s),
                    float(row.R_mean),
                    s=85,
                    color=CONFIG_COLORS[str(row.config)],
                    marker=DOMAIN_MARKERS[domain],
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.annotate(
                    CONFIG_SHORT[str(row.config)],
                    (float(row.latency_s), float(row.R_mean)),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color="#222222",
                )

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean end-to-end latency (s)")
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"Audited certified reliability $R$")

    config_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CONFIG_COLORS[c], markeredgecolor="#222222", label=CONFIG_LABELS[c], markersize=7)
        for c in CONFIG_ORDER
    ]
    domain_handles = [
        Line2D([0], [0], marker=DOMAIN_MARKERS[d], color=DOMAIN_COLORS[d], label=DOMAIN_LABELS[d], linestyle="none", markersize=7)
        for d in DOMAIN_ORDER
    ]
    fig.legend(
        handles=config_handles + domain_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=8,
        frameon=False,
    )
    return _save_figure(fig, output_path)


def make_result_plots(
    *,
    tables_dir: str | Path = "outputs/tables/final_a6_dual_model_full_v2",
    output_dir: str | Path = "figures",
    prefix: str = "final_a6",
) -> list[Path]:
    tables_root = Path(tables_dir)
    out_root = Path(output_dir)

    significance_df = _read_csv(tables_root / "significance.csv")
    frontier_df = _read_csv(tables_root / "cost_frontier.csv").merge(
        _read_csv(tables_root / "latency.csv")[["domain", "model", "config", "latency_avg_ms"]],
        on=["domain", "model", "config"],
        how="inner",
    )

    outputs = [
        _make_full_vs_none_delta_plot(significance_df, out_root / f"{prefix}_full_vs_none_delta.pdf"),
        _make_latency_operating_points_plot(frontier_df, out_root / f"{prefix}_latency_operating_points.pdf"),
    ]
    return outputs
