from __future__ import annotations

from typing import Any, Iterable
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    RESULTS_GRAPHS_DIR,
    OUT_ETHNIC_WEALTH_GAPS_MAX_PDF,
    OUT_ETHNIC_WEALTH_GAPS_MAX_TIFF,
    OUT_ETHNIC_WEALTH_GAPS_MAX_PNG,
    OUT_ETHNIC_WEALTH_GAPS_AVG_PDF,
    OUT_ETHNIC_WEALTH_GAPS_AVG_TIFF,
    OUT_ETHNIC_WEALTH_GAPS_AVG_PNG,
    OUT_ETHNIC_WEALTH_GAPS_GINI_PDF,
    OUT_ETHNIC_WEALTH_GAPS_GINI_TIFF,
    OUT_ETHNIC_WEALTH_GAPS_GINI_PNG,
    OUT_RELATIVE_REPRESENTATION_POOREST_PDF,
    OUT_RELATIVE_REPRESENTATION_POOREST_TIFF,
    OUT_RELATIVE_REPRESENTATION_POOREST_PNG,
    OUT_RELATIVE_REPRESENTATION_RICHEST_PDF,
    OUT_RELATIVE_REPRESENTATION_RICHEST_TIFF,
    OUT_RELATIVE_REPRESENTATION_RICHEST_PNG,
    OUT_SPINE_PLOT_PDF,
    OUT_SPINE_PLOT_TIFF,
    OUT_SPINE_PLOT_PNG,
    OUT_HEATMAP_PDF,
    OUT_HEATMAP_TIFF,
    OUT_HEATMAP_PNG,
    WEALTH_ORDER,
)


# Keys added later in the pipeline that are not country-level analysis objects.
SUMMARY_KEYS = {
    "datapoints",
    "num_ethnicities_per_country",
    "ethnicities_retained",
}

# Centralized plotting defaults for consistent publication-ready output.
PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}

BAR_COLOR = "#7f7f7f"
GRID_COLOR = "0.88"
SPINE_PLOT_COLORS = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#636363"]


def _apply_publication_style() -> None:
    """Apply a consistent plotting style across all exported figures."""
    mpl.rcParams.update(PUBLICATION_RCPARAMS)


def _iter_country_items(data: dict[str, dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield only country entries and skip pipeline-level summary objects."""
    for country, info in data.items():
        if country not in SUMMARY_KEYS:
            yield country, info


def _finalize_axes(ax: plt.Axes, *, add_y_grid: bool = True) -> None:
    """Apply shared axis formatting used by most figures."""
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if add_y_grid:
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color=GRID_COLOR)
        ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, pdf_path, tiff_path, png_path) -> None:
    """Save one figure in PDF, TIFF, and PNG formats and close it."""
    RESULTS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(png_path, dpi=600)
    plt.close(fig)


def _build_country_bar_plot(
    summary: pd.DataFrame,
    value_col: str,
    error_col: str,
    ylabel: str,
    title: str,
    *,
    figsize: tuple[float, float] = (10.5, 6.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a standardized bar plot for country-level summaries."""
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        summary["country"],
        summary[value_col],
        yerr=summary[error_col],
        capsize=3,
        width=0.72,
        color=BAR_COLOR,
        linewidth=0.7,
        error_kw={"ecolor": "0.3", "elinewidth": 1.0, "capsize": 3},
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["country"], rotation=45, ha="right")
    _finalize_axes(ax)
    return fig, ax


def _safe_filename(text: str) -> str:
    """Convert a country name into a filesystem-safe filename fragment."""
    text = str(text).strip().replace("’", "'")
    text = re.sub(r"[^\w\s\-']", "_", text)
    return re.sub(r"\s+", "_", text)


def _compute_ratio_ci95(national_prop: Any, se_diff: Any) -> float:
    """Approximate a 95% CI width for a representation ratio."""
    if national_prop is None or pd.isna(national_prop) or national_prop == 0:
        return np.nan
    if se_diff is None or pd.isna(se_diff):
        return np.nan
    return float(1.96 * (se_diff / national_prop))


def gini(array: np.ndarray) -> float:
    """Return the Gini coefficient of a numeric array.

    The function returns NaN for empty arrays or arrays with mean zero.
    """
    x = np.asarray(array, dtype=float)
    if x.size == 0:
        return np.nan

    x = np.sort(x)
    mu = x.mean()
    if mu == 0:
        return np.nan

    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    n = x.size
    return float(diff_sum / (2 * n**2 * mu))


def graphs_wealth_gaps_max(data: dict[str, dict[str, Any]]) -> None:
    """Plot the maximum absolute pairwise ethnic wealth gap for each country."""
    records: list[dict[str, Any]] = []

    for country, info in _iter_country_items(data):
        pairwise_df = pd.DataFrame(info.get("pairwise_results"))
        if pairwise_df.empty:
            continue

        pairwise_df = pairwise_df.copy()
        pairwise_df["abs_gap_mean"] = pairwise_df["gap_mean"].abs()
        max_row = pairwise_df.loc[pairwise_df["abs_gap_mean"].idxmax()]
        records.append(
            {
                "country": country,
                "max_gap": float(max_row["abs_gap_mean"]),
                "ci95": float(1.96 * max_row["gap_se"]),
            }
        )

    summary = pd.DataFrame(records).sort_values("max_gap", ascending=False).reset_index(drop=True)
    fig, _ = _build_country_bar_plot(
        summary,
        value_col="max_gap",
        error_col="ci95",
        ylabel="Maximum absolute pairwise difference in mean wealth quintile rank",
        title="Maximum pairwise ethnic wealth gaps across countries (95% CI)",
    )
    _save_figure(fig, OUT_ETHNIC_WEALTH_GAPS_MAX_PDF, OUT_ETHNIC_WEALTH_GAPS_MAX_TIFF, OUT_ETHNIC_WEALTH_GAPS_MAX_PNG)


def graphs_wealth_gaps_avg(data: dict[str, dict[str, Any]]) -> None:
    """Plot the average absolute pairwise ethnic wealth gap for each country."""
    records: list[dict[str, Any]] = []

    for country, info in _iter_country_items(data):
        pairwise_df = pd.DataFrame(info.get("pairwise_results"))
        if pairwise_df.empty:
            continue

        abs_gaps = pairwise_df["gap_mean"].abs()
        records.append(
            {
                "country": country,
                "avg_gap": float(abs_gaps.mean()),
                "std_gap": float(abs_gaps.std()),
            }
        )

    summary = pd.DataFrame(records).sort_values("avg_gap", ascending=False).reset_index(drop=True)
    fig, _ = _build_country_bar_plot(
        summary,
        value_col="avg_gap",
        error_col="std_gap",
        ylabel="Average absolute pairwise difference in mean wealth quintile rank",
        title="Average pairwise ethnic wealth gaps across countries (±1 SD)",
    )
    _save_figure(fig, OUT_ETHNIC_WEALTH_GAPS_AVG_PDF, OUT_ETHNIC_WEALTH_GAPS_AVG_TIFF, OUT_ETHNIC_WEALTH_GAPS_AVG_PNG)


def graphs_wealth_gaps_gini(data: dict[str, dict[str, Any]]) -> None:
    """Plot the Gini coefficient of ethnic mean wealth ranks for each country."""
    records: list[dict[str, Any]] = []
    bootstrap_draws = 5000
    rng = np.random.default_rng(12345)

    for country, info in _iter_country_items(data):
        mean_rank_df = pd.DataFrame(info.get("mean_quintiles"))
        if mean_rank_df.empty:
            continue

        values = mean_rank_df["mean_quintile_rank"].to_numpy(float)
        ci95 = np.nan
        cov_df = info.get("cov_mean_quintile_rank")

        if cov_df is not None:
            try:
                ethnicity_order = mean_rank_df["ethnicity"].tolist()
                cov_matrix = cov_df.loc[ethnicity_order, ethnicity_order].to_numpy(float)
                draws = rng.multivariate_normal(mean=values, cov=cov_matrix, size=bootstrap_draws)
                gini_draws = np.apply_along_axis(gini, 1, draws)
                gini_draws = gini_draws[np.isfinite(gini_draws)]
                if gini_draws.size > 1:
                    ci95 = float(1.96 * np.std(gini_draws, ddof=1))
            except Exception:
                # Fall back to the point estimate if covariance-based simulation fails.
                ci95 = np.nan

        records.append(
            {
                "country": country,
                "gini_ethnic_inequality": gini(values),
                "ci95": ci95,
            }
        )

    summary = (
        pd.DataFrame(records)
        .sort_values("gini_ethnic_inequality", ascending=False)
        .reset_index(drop=True)
    )
    fig, _ = _build_country_bar_plot(
        summary,
        value_col="gini_ethnic_inequality",
        error_col="ci95",
        ylabel="Gini coefficient of mean wealth quintile rank across ethnic groups",
        title="Gini coefficients of ethnic wealth inequality across countries (95% CI)",
    )
    _save_figure(fig, OUT_ETHNIC_WEALTH_GAPS_GINI_PDF, OUT_ETHNIC_WEALTH_GAPS_GINI_TIFF, OUT_ETHNIC_WEALTH_GAPS_GINI_PNG)


def spine_plot_country(
    country: str,
    df: pd.DataFrame,
    i: int,
    wealth_col: str = "wealth_index",
    eth_col: str = "ethnicity",
    wealth_order: tuple[str, ...] = ("poorest", "poorer", "middle", "richer", "richest"),
) -> None:
    """Create one spine plot showing wealth-quintile composition within ethnicity."""
    plot_df = df[[eth_col, wealth_col]].dropna(subset=[eth_col, wealth_col]).copy()
    crosstab = pd.crosstab(plot_df[eth_col], plot_df[wealth_col])

    for category in wealth_order:
        if category not in crosstab.columns:
            crosstab[category] = 0
    crosstab = crosstab[list(wealth_order)]

    row_totals = crosstab.sum(axis=1).astype(float).replace(0, np.nan)
    valid_rows = row_totals.notna()
    crosstab = crosstab.loc[valid_rows]
    row_totals = row_totals.loc[valid_rows]
    proportions = crosstab.div(row_totals, axis=0).fillna(0.0)

    widths = row_totals / row_totals.sum()
    positions = widths.cumsum() - widths
    centers = positions + (widths / 2)

    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    bottoms = np.zeros(len(proportions), dtype=float)
    for idx, category in enumerate(proportions.columns):
        heights = proportions[category].to_numpy(float)
        ax.bar(
            x=positions.to_numpy(float),
            height=heights,
            width=widths.to_numpy(float),
            bottom=bottoms,
            align="edge",
            label=category,
            color=SPINE_PLOT_COLORS[idx % len(SPINE_PLOT_COLORS)],
            edgecolor="black",
            linewidth=0.6,
        )
        bottoms += heights

    ax.set_xticks(centers.to_numpy(float))
    ax.set_xticklabels(proportions.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("Share within ethnicity")
    ax.set_title(f"Wealth distribution by ethnicity: {country}")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    _finalize_axes(ax)
    ax.legend(title="Wealth quintile", frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")

    safe_country = _safe_filename(country)
    _save_figure(
        fig,
        OUT_SPINE_PLOT_PDF.with_name(OUT_SPINE_PLOT_PDF.name.format(i=i, country=safe_country)),
        OUT_SPINE_PLOT_TIFF.with_name(OUT_SPINE_PLOT_TIFF.name.format(i=i, country=safe_country)),
        OUT_SPINE_PLOT_PNG.with_name(OUT_SPINE_PLOT_PNG.name.format(i=i, country=safe_country)),
    )


def graphs_spine_plots(
    data: dict[str, dict[str, Any]],
    wealth_col: str = "wealth_index",
    eth_col: str = "ethnicity",
    wealth_order: tuple[str, ...] = ("poorest", "poorer", "middle", "richer", "richest"),
) -> None:
    """Create one spine plot per country."""
    for i, (country, info) in enumerate(_iter_country_items(data), start=1):
        df = info.get("dataset")
        if df is None or df.empty:
            continue
        spine_plot_country(
            country=country,
            df=df,
            i=i,
            wealth_col=wealth_col,
            eth_col=eth_col,
            wealth_order=wealth_order,
        )


def heatmap_country(
    df: pd.DataFrame,
    country: str,
    i: int,
    distribution_wealth: pd.Series | pd.DataFrame | dict[str, float] | None = None,
    weight_col: str = "sampling_weight_women",
    eth_col: str = "ethnicity",
    wealth_col: str = "wealth_index",
) -> None:
    """Create a weighted ethnicity-by-wealth heatmap for one country.

    Rows represent ethnic groups, columns represent wealth quintiles, and values
    are row percentages within each ethnicity.
    """
    plot_df = df[[eth_col, wealth_col, weight_col]].dropna(subset=[eth_col, wealth_col, weight_col]).copy()

    weighted_table = pd.pivot_table(
        plot_df,
        index=eth_col,
        columns=wealth_col,
        values=weight_col,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )

    for category in WEALTH_ORDER:
        if category not in weighted_table.columns:
            weighted_table[category] = 0
    weighted_table = weighted_table[WEALTH_ORDER]

    if distribution_wealth is not None:
        if isinstance(distribution_wealth, pd.DataFrame):
            national_series = distribution_wealth.squeeze()
        elif isinstance(distribution_wealth, dict):
            national_series = pd.Series(distribution_wealth)
        else:
            national_series = distribution_wealth.copy()

        national_series = pd.Series(national_series).reindex(WEALTH_ORDER).fillna(0)
        weighted_table.loc["National"] = national_series.values
        weighted_table = weighted_table.loc[[idx for idx in weighted_table.index if idx != "National"] + ["National"]]

    row_sums = weighted_table.sum(axis=1).replace(0, np.nan)
    percentages = (weighted_table.div(row_sums, axis=0) * 100).fillna(0).round(1)

    _apply_publication_style()
    fig_height = max(4.5, 0.35 * len(percentages.index) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    image = ax.imshow(
        percentages.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(np.arange(len(percentages.columns)))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_yticks(np.arange(len(percentages.index)))
    ax.set_yticklabels(percentages.index.tolist())
    ax.set_xlabel("Wealth quintile")
    ax.set_ylabel("Ethnicity")
    ax.set_title(f"Wealth distribution by ethnicity: {country}")
    _finalize_axes(ax, add_y_grid=False)

    values = percentages.to_numpy()
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 50 else "black",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share within ethnicity (%)")

    safe_country = _safe_filename(country)
    _save_figure(
        fig,
        OUT_HEATMAP_PDF.with_name(OUT_HEATMAP_PDF.name.format(i=i, country=safe_country)),
        OUT_HEATMAP_TIFF.with_name(OUT_HEATMAP_TIFF.name.format(i=i, country=safe_country)),
        OUT_HEATMAP_PNG.with_name(OUT_HEATMAP_PNG.name.format(i=i, country=safe_country)),
    )


def graphs_heatmaps(
    data: dict[str, dict[str, Any]],
    weight_col: str = "sampling_weight_women",
    eth_col: str = "ethnicity",
    wealth_col: str = "wealth_index",
) -> None:
    """Create one weighted heatmap per country."""
    for i, (country, info) in enumerate(_iter_country_items(data), start=1):
        df = info.get("dataset")
        if df is None or df.empty:
            continue
        heatmap_country(
            df=df,
            country=country,
            i=i,
            distribution_wealth=info.get("distribution_wealth"),
            weight_col=weight_col,
            eth_col=eth_col,
            wealth_col=wealth_col,
        )


def _representation_plot(
    data: dict[str, dict[str, Any]],
    *,
    subgroup_key: str,
    national_prop_key: str,
    ratio_col: str,
    ci_col: str,
    title: str,
    output_pdf,
    output_tiff,
    output_png,
) -> None:
    """Create a country-by-group representation-ratio bar chart."""
    records: list[dict[str, Any]] = []

    for country, info in _iter_country_items(data):
        subgroup = info.get(subgroup_key)
        if not subgroup:
            continue

        ethnicity = subgroup.get("ethnicity")
        ratio = subgroup.get("ratio")
        national_prop = subgroup.get(national_prop_key)
        se_diff = subgroup.get("se_diff")
        records.append(
            {
                "country": country,
                "ethnicity": ethnicity,
                "country_eth": f"{country}: {ethnicity}",
                ratio_col: float(ratio),
                ci_col: _compute_ratio_ci95(national_prop, se_diff),
            }
        )

    summary = pd.DataFrame(records).sort_values(ratio_col, ascending=False).reset_index(drop=True)
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        summary["country_eth"],
        summary[ratio_col],
        yerr=summary[ci_col],
        capsize=4,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
        error_kw={"ecolor": "0.3", "lw": 1.0},
    )
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["country_eth"], rotation=90)
    ax.set_ylabel("Representation ratio (group vs. national average)")
    ax.set_title(title)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
    _finalize_axes(ax)
    _save_figure(fig, output_pdf, output_tiff, output_png)


def graphs_relative_representation_poorest(data: dict[str, dict[str, Any]]) -> None:
    """Plot the over- or under-representation of the poorest ethnic group in Q1."""
    _representation_plot(
        data,
        subgroup_key="poorest_ethnicity",
        national_prop_key="prop_poorest_quintile_national_weighted",
        ratio_col="ratio_poorest",
        ci_col="ci95_ratio_poorest",
        title="Representation of the poorest ethnic group in the poorest wealth quintile (95% CI)",
        output_pdf=OUT_RELATIVE_REPRESENTATION_POOREST_PDF,
        output_tiff=OUT_RELATIVE_REPRESENTATION_POOREST_TIFF,
        output_png=OUT_RELATIVE_REPRESENTATION_POOREST_PNG,
    )


def graphs_relative_representation_richest(data: dict[str, dict[str, Any]]) -> None:
    """Plot the over- or under-representation of the wealthiest ethnic group in Q5."""
    _representation_plot(
        data,
        subgroup_key="richest_ethnicity",
        national_prop_key="prop_richest_quintile_national_weighted",
        ratio_col="ratio_richest",
        ci_col="ci95_ratio_richest",
        title="Representation of the wealthiest ethnic group in the richest wealth quintile (95% CI)",
        output_pdf=OUT_RELATIVE_REPRESENTATION_RICHEST_PDF,
        output_tiff=OUT_RELATIVE_REPRESENTATION_RICHEST_TIFF,
        output_png=OUT_RELATIVE_REPRESENTATION_RICHEST_PNG,
    )


__all__ = [
    "gini",
    "graphs_wealth_gaps_max",
    "graphs_wealth_gaps_avg",
    "graphs_wealth_gaps_gini",
    "spine_plot_country",
    "graphs_spine_plots",
    "heatmap_country",
    "graphs_heatmaps",
    "graphs_relative_representation_poorest",
    "graphs_relative_representation_richest",
]
