from __future__ import annotations

"""Analysis routines for wealth-by-ethnicity comparisons.

This module groups the core analytic steps used by ``pipeline.py`` into four
public functions with a stable API:

- ``compute_weighted_wealth_distribution``
- ``compute_weighted_crosstabs``
- ``compute_mean_rank_results``
- ``survey_prop_diff_test``

Each public function mutates the shared ``data`` dictionary in place. This
behavior is preserved because the surrounding pipeline expects these side
effects.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm as ndist
from statsmodels.stats.multitest import multipletests

from .config import ALPHA, QCOLS, QRANK, WEALTH_ORDER

DataDict = dict[str, dict[str, Any]]
CountryInfo = dict[str, Any]


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def compute_weighted_wealth_distribution(data: DataDict) -> None:
    """Compute weighted national wealth-quintile distributions.

    For each country, the function calculates the weighted percentage of women
    in each wealth quintile and stores the result under
    ``info["distribution_wealth"]``.
    """
    for info in data.values():
        df = info["dataset"]

        distribution = (
            df.groupby("wealth_index", observed=True)["sampling_weight_women"]
            .sum()
            .div(df["sampling_weight_women"].sum())
            .mul(100)
            .round(2)
            .reindex(WEALTH_ORDER)
        )

        info["distribution_wealth"] = distribution


def compute_weighted_crosstabs(data: DataDict) -> None:
    """Compute weighted ethnicity-by-wealth tables for each country.

    For each country, this function stores:

    - ``crosstab``: weighted totals by wealth quintile and ethnicity
    - ``distribution_across_quintiles``: within-ethnicity percentages across
      wealth quintiles
    """
    for info in data.values():
        df = info["dataset"]

        weighted_table = (
            df.groupby(["wealth_index", "ethnicity"], observed=True)["sampling_weight_women"]
            .sum()
            .unstack(fill_value=0)
        )

        available_rows = [quintile for quintile in WEALTH_ORDER if quintile in weighted_table.index]
        if available_rows:
            weighted_table = weighted_table.reindex(available_rows)

        column_totals = weighted_table.sum(axis=0).replace(0, pd.NA)
        distribution = weighted_table.div(column_totals, axis=1).mul(100).round(2)

        info["crosstab"] = weighted_table
        info["distribution_across_quintiles"] = distribution


def compute_mean_rank_results(data: DataDict) -> None:
    """Compute ethnicity-level mean rank results and associated tests.

    For each country, the function:

    1. creates a rank score from the wealth-quintile dummy columns,
    2. computes weighted mean rank statistics by ethnicity,
    3. estimates the covariance matrix using PSU-within-stratum clustering,
    4. performs pairwise ethnicity comparisons with Holm correction, and
    5. performs a global Wald test across all ethnic groups.

    Results are stored in place under the existing country entry.
    """
    for country, info in data.items():
        df = info["dataset"].copy()

        df = _prepare_rank_data(df)
        group_stats, ethnicities = _compute_ethnicity_means(df)
        covariance_matrix, df = _compute_ethnicity_covariance(df, group_stats, ethnicities)

        info.update(
            {
                "dataset": df,
                "mean_quintiles": _build_mean_rank_results(group_stats, ethnicities, covariance_matrix),
                "cov_mean_quintile_rank": pd.DataFrame(
                    covariance_matrix,
                    index=ethnicities,
                    columns=ethnicities,
                ),
                "global_test": _compute_global_wald_test(
                    mean_vector=group_stats["mean_quintile_rank"].to_numpy(float),
                    covariance_matrix=covariance_matrix,
                ),
                "pairwise_results": _compute_pairwise_mean_rank_tests(
                    country=country,
                    group_stats=group_stats,
                    ethnicities=ethnicities,
                    covariance_matrix=covariance_matrix,
                ),
            }
        )


def survey_prop_diff_test(data: DataDict) -> None:
    """Compare extreme-group quintile proportions with national proportions.

    For each country, the function identifies the poorest and richest
    ethnicities from ``mean_quintiles`` and compares their weighted proportion
    in the poorest or richest wealth quintile to the corresponding national
    weighted proportion using a design-based Wald test.

    Results are stored in place under:

    - ``poorest_ethnicity``
    - ``richest_ethnicity``
    """
    for info in data.values():
        df = info["dataset"]
        mean_quintiles = info["mean_quintiles"]

        poorest_ethnicity = mean_quintiles.loc[
            mean_quintiles["mean_quintile_rank"].idxmin(), "ethnicity"
        ]
        richest_ethnicity = mean_quintiles.loc[
            mean_quintiles["mean_quintile_rank"].idxmax(), "ethnicity"
        ]

        info["poorest_ethnicity"] = _build_extreme_group_result(
            df=df,
            ethnicity=poorest_ethnicity,
            wealth_col="wealth_poorest",
            quintile_label="poorest",
        )
        info["richest_ethnicity"] = _build_extreme_group_result(
            df=df,
            ethnicity=richest_ethnicity,
            wealth_col="wealth_richest",
            quintile_label="richest",
        )


# ---------------------------------------------------------------------------
# Rank-based analysis helpers
# ---------------------------------------------------------------------------

def _prepare_rank_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank-based columns used in the ethnicity mean-rank analysis."""
    df["q_rank"] = df[QCOLS].to_numpy() @ QRANK
    df["wy"] = df["sampling_weight_women"] * df["q_rank"]
    return df


def _compute_ethnicity_means(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute weighted mean rank statistics and effective sample sizes."""
    group_stats = df.groupby("ethnicity", observed=True).agg(
        W=("sampling_weight_women", "sum"),
        Wy=("wy", "sum"),
        n_unweighted=("q_rank", "size"),
        sum_w2=("sampling_weight_women", lambda s: np.square(s.astype(float)).sum()),
    )

    group_stats["mean_quintile_rank"] = group_stats["Wy"] / group_stats["W"]
    group_stats["n_eff"] = group_stats["W"] ** 2 / group_stats["sum_w2"]

    ethnicities = group_stats["mean_quintile_rank"].sort_values().index.to_list()
    group_stats = group_stats.loc[ethnicities]

    return group_stats, ethnicities


def _compute_ethnicity_covariance(
    df: pd.DataFrame,
    group_stats: pd.DataFrame,
    ethnicities: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate the covariance matrix of ethnicity-specific mean ranks."""
    df["ethnicity"] = pd.Categorical(df["ethnicity"], categories=ethnicities, ordered=True)

    ranks = df["q_rank"].to_numpy(float)
    weights = df["sampling_weight_women"].to_numpy(float)
    means = group_stats["mean_quintile_rank"].to_numpy(float)
    weight_totals = group_stats["W"].to_numpy(float)
    ethnicity_codes = df["ethnicity"].cat.codes.to_numpy()

    influence_columns: list[str] = []
    for group_index, _ in enumerate(ethnicities):
        column_name = f"IF_{group_index}"
        df[column_name] = (
            weights
            * (ranks - means[group_index])
            * (ethnicity_codes == group_index)
            / weight_totals[group_index]
        )
        influence_columns.append(column_name)

    covariance_matrix = _cluster_covariance_by_stratum(
        df=df,
        value_columns=influence_columns,
        stratum_col="stratum",
        psu_col="PSU",
    )

    return covariance_matrix, df


def _build_mean_rank_results(
    group_stats: pd.DataFrame,
    ethnicities: list[str],
    covariance_matrix: np.ndarray,
) -> pd.DataFrame:
    """Assemble the ethnicity-level summary table for mean rank results."""
    standard_errors = np.sqrt(np.diag(covariance_matrix))

    return pd.DataFrame(
        {
            "ethnicity": ethnicities,
            "mean_quintile_rank": group_stats["mean_quintile_rank"].to_numpy(float),
            "se_mean_quintile_rank": standard_errors,
            "n_unweighted": group_stats["n_unweighted"].to_numpy(),
            "n_eff": group_stats["n_eff"].to_numpy(float),
        }
    )


def _compute_pairwise_mean_rank_tests(
    country: str,
    group_stats: pd.DataFrame,
    ethnicities: list[str],
    covariance_matrix: np.ndarray,
) -> pd.DataFrame:
    """Perform pairwise tests of ethnicity differences in mean rank."""
    n_groups = len(ethnicities)
    if n_groups < 2:
        return pd.DataFrame()

    means = group_stats["mean_quintile_rank"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            difference = means[i] - means[j]
            variance = max(
                covariance_matrix[i, i]
                + covariance_matrix[j, j]
                - 2 * covariance_matrix[i, j],
                0.0,
            )
            standard_error = float(np.sqrt(variance))

            if standard_error <= 0 or np.isnan(standard_error):
                z_value = np.nan
                p_value = np.nan
            else:
                z_value = difference / standard_error
                p_value = 2.0 * (1.0 - ndist.cdf(abs(z_value)))

            rows.append(
                {
                    "country": country,
                    "ethnicity_i": ethnicities[i],
                    "ethnicity_j": ethnicities[j],
                    "gap_mean": difference,
                    "gap_se": standard_error,
                    "z": z_value,
                    "p_raw": p_value,
                }
            )
            raw_p_values.append(p_value)

    adjusted_p_values = _holm_adjust(raw_p_values)

    for row, p_adjusted in zip(rows, adjusted_p_values):
        row["p_holm"] = p_adjusted
        row["significant_0.05"] = bool(p_adjusted < ALPHA) if not np.isnan(p_adjusted) else False

    return pd.DataFrame(rows)


def _compute_global_wald_test(
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
) -> dict[str, float | int]:
    """Run a global Wald test for equality of ethnicity mean ranks."""
    n_groups = len(mean_vector)
    if n_groups <= 1:
        return {"statistic": np.nan, "df": 0, "p_value": np.nan}

    contrast_matrix = np.eye(n_groups - 1, n_groups)
    contrast_matrix[:, -1] = -1.0

    mean_differences = contrast_matrix @ mean_vector
    covariance_differences = contrast_matrix @ covariance_matrix @ contrast_matrix.T

    statistic = float(
        mean_differences.T @ np.linalg.pinv(covariance_differences) @ mean_differences
    )
    df_wald = n_groups - 1
    p_value = float(1.0 - chi2.cdf(statistic, df_wald))

    return {"statistic": statistic, "df": df_wald, "p_value": p_value}


# ---------------------------------------------------------------------------
# Proportion comparison helpers
# ---------------------------------------------------------------------------

def _build_extreme_group_result(
    df: pd.DataFrame,
    ethnicity: str,
    wealth_col: str,
    quintile_label: str,
) -> dict[str, Any]:
    """Format the output for the poorest or richest ethnicity comparison."""
    result = _compute_prop_comparison(
        df=df,
        target_ethnicity=ethnicity,
        wealth_col=wealth_col,
    )

    return {
        "ethnicity": ethnicity,
        f"prop_{quintile_label}_quintile_ethnicity_weighted": result["p_ethnicity_weighted"],
        f"prop_{quintile_label}_quintile_national_weighted": result["p_national_weighted"],
        "ratio": result["ratio"],
        "difference": result["diff"],
        "se_diff": result["se_diff"],
        "z_diff": result["z"],
        "p_value_diff": result["p_value"],
        "ci95_lower_diff": result["ci_lower"],
        "ci95_upper_diff": result["ci_upper"],
    }


def _compute_prop_comparison(
    df: pd.DataFrame,
    target_ethnicity: str,
    wealth_col: str,
    eth_col: str = "ethnicity",
    stratum_col: str = "stratum",
    psu_col: str = "PSU",
    weight_col: str = "sampling_weight_women",
) -> dict[str, float]:
    """Compare an ethnicity-specific weighted proportion with the national one."""
    columns = [weight_col, wealth_col, eth_col, stratum_col, psu_col]
    working_df = df[columns].copy()

    weights = working_df[weight_col].astype(float).to_numpy()
    outcome = working_df[wealth_col].astype(float).to_numpy()
    is_target_ethnicity = (working_df[eth_col] == target_ethnicity).astype(float).to_numpy()

    # Weighted national proportion.
    national_weight_total = weights.sum()
    p_national = (weights * outcome).sum() / national_weight_total

    # Weighted ethnicity-specific proportion.
    ethnicity_mask = is_target_ethnicity > 0
    ethnicity_weights = weights[ethnicity_mask]
    ethnicity_outcome = outcome[ethnicity_mask]
    ethnicity_weight_total = ethnicity_weights.sum()
    p_ethnicity = (ethnicity_weights * ethnicity_outcome).sum() / ethnicity_weight_total

    ratio = p_ethnicity / p_national if p_national != 0 else np.nan

    # Influence-function based variance estimation.
    working_df["if_ethnicity"] = (
        weights * is_target_ethnicity * (outcome - p_ethnicity) / ethnicity_weight_total
    )
    working_df["if_national"] = weights * (outcome - p_national) / national_weight_total

    covariance_matrix = _cluster_covariance_by_stratum(
        df=working_df,
        value_columns=["if_ethnicity", "if_national"],
        stratum_col=stratum_col,
        psu_col=psu_col,
    )

    var_ethnicity = covariance_matrix[0, 0]
    var_national = covariance_matrix[1, 1]
    cov_ethnicity_national = covariance_matrix[0, 1]

    difference = p_ethnicity - p_national
    variance_difference = var_ethnicity + var_national - 2.0 * cov_ethnicity_national

    if variance_difference <= 0 or np.isnan(variance_difference):
        se_difference = np.nan
        z_value = np.nan
        p_value = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        se_difference = float(np.sqrt(variance_difference))
        z_value = float(difference / se_difference)
        p_value = float(2.0 * (1.0 - ndist.cdf(abs(z_value))))
        ci_lower = float(difference - 1.96 * se_difference)
        ci_upper = float(difference + 1.96 * se_difference)

    return {
        "p_ethnicity_weighted": float(p_ethnicity),
        "p_national_weighted": float(p_national),
        "ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "diff": float(difference),
        "se_diff": float(se_difference) if np.isfinite(se_difference) else np.nan,
        "z": float(z_value) if np.isfinite(z_value) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "ci_lower": float(ci_lower) if np.isfinite(ci_lower) else np.nan,
        "ci_upper": float(ci_upper) if np.isfinite(ci_upper) else np.nan,
    }


# ---------------------------------------------------------------------------
# Shared statistical helpers
# ---------------------------------------------------------------------------

def _cluster_covariance_by_stratum(
    df: pd.DataFrame,
    value_columns: list[str],
    stratum_col: str,
    psu_col: str,
) -> np.ndarray:
    """Estimate a covariance matrix from clustered sums within strata."""
    cluster_sums = df.groupby([stratum_col, psu_col], observed=True)[value_columns].sum()

    covariance_matrix = np.zeros((len(value_columns), len(value_columns)), dtype=float)

    for _, block in cluster_sums.groupby(level=0, observed=False):
        cluster_values = block.to_numpy()
        n_clusters = len(cluster_values)

        # A stratum with one PSU contributes no within-stratum variance.
        if n_clusters <= 1:
            continue

        centered = cluster_values - cluster_values.mean(axis=0, keepdims=True)
        covariance_matrix += (n_clusters / (n_clusters - 1.0)) * (centered.T @ centered)

    return covariance_matrix


def _holm_adjust(p_values: list[float]) -> np.ndarray:
    """Apply Holm correction while preserving missing values."""
    p_values_array = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p_values_array)

    adjusted = np.full_like(p_values_array, np.nan)
    if valid.any():
        _, adjusted_valid, _, _ = multipletests(
            p_values_array[valid],
            alpha=ALPHA,
            method="holm",
        )
        adjusted[valid] = adjusted_valid

    return adjusted
