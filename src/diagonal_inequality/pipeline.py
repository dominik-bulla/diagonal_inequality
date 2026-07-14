from __future__ import annotations

"""Top-level analysis pipeline for DHS and MICS wealth-by-ethnicity analyses.

This module orchestrates the full workflow used in the project:

1. Create output directories.
2. Load DHS and MICS datasets.
3. Apply cleaning and filtering steps.
4. Run analysis routines.
5. Export tables and figures.

The public entry point, :func:`run_pipeline`, is intentionally kept stable
because it may be imported and called from elsewhere in the project.
"""

from typing import Any

from .analysis import (
    compute_mean_rank_results,
    compute_weighted_crosstabs,
    compute_weighted_wealth_distribution,
    survey_prop_diff_test,
)
from .cleaning import (
    add_dummies_and_scale_weights,
    clean_ethnicity_labels,
    drop_missing_wealth_index,
    export_all_ethnicities,
    retain_large_ethnic_groups,
    validate_final_data,
)
from .config import MIN_ETHNICITY_N
from .export_graphs import (
    graphs_heatmaps,
    graphs_relative_representation_poorest,
    graphs_relative_representation_richest,
    graphs_spine_plots,
    graphs_wealth_gaps_avg,
    graphs_wealth_gaps_gini,
    graphs_wealth_gaps_max,
)
from .export_tables import (
    ensure_directories,
    table_country_overviews,
    table_distribution_across_quintiles,
    table_distribution_wealth,
    table_ethnicities_retained,
    table_mean_quintiles,
    table_num_ethnicities_per_country,
    table_pairwise_results,
)
from .io import load_dhs, load_mics

DataDict = dict[str, Any]


def _load_and_merge_data() -> tuple[DataDict, Any, Any]:
    """Load DHS and MICS data and return one combined country-sorted mapping."""
    data_mics = load_mics()
    data_dhs = load_dhs()

    combined_data = {**data_dhs, **data_mics}
    combined_data = {country: combined_data[country] for country in sorted(combined_data)}
    return combined_data



def _clean_and_prepare_data(data: DataDict) -> tuple[Any, Any, Any]:
    """Apply all cleaning and feature-preparation steps in pipeline order."""
    # Remove rows without valid wealth information.
    drop_missing_wealth_index(data)

    # Harmonize ethnicity labels and retain only sufficiently large groups.
    clean_ethnicity_labels(data)
    country_overviews = export_all_ethnicities(data)
    num_ethnicities_per_country, ethnicities_retained = retain_large_ethnic_groups(
        data,
        min_n=MIN_ETHNICITY_N,
    )

    # Final feature construction and validation before analysis.
    add_dummies_and_scale_weights(data)
    validate_final_data(data)

    return country_overviews, num_ethnicities_per_country, ethnicities_retained



def _run_analyses(data: DataDict) -> None:
    """Run all analytic routines that populate results in-place."""
    compute_weighted_wealth_distribution(data)
    compute_weighted_crosstabs(data)
    compute_mean_rank_results(data)
    survey_prop_diff_test(data)



def _export_tables(
    data: DataDict,
    country_overviews: Any,
    num_ethnicities_per_country: Any,
    ethnicities_retained: Any,
) -> None:
    """Export all project tables."""
    table_country_overviews(country_overviews)
    table_num_ethnicities_per_country(num_ethnicities_per_country)
    table_ethnicities_retained(ethnicities_retained)
    table_distribution_wealth(data)
    table_distribution_across_quintiles(data)
    table_mean_quintiles(data)
    table_pairwise_results(data)



def _export_graphs(data: DataDict) -> None:
    """Export all project figures."""
    graphs_wealth_gaps_max(data)
    graphs_wealth_gaps_avg(data)
    graphs_wealth_gaps_gini(data)
    graphs_spine_plots(data)
    graphs_heatmaps(data)
    graphs_relative_representation_poorest(data)
    graphs_relative_representation_richest(data)



def run_pipeline() -> dict[str, Any]:
    """Run the full data-processing, analysis, and export pipeline.

    Returns
    -------
    dict[str, Any]
        The processed per-country data objects together with key summary
        tables generated during cleaning.
    """
    ensure_directories()

    data = _load_and_merge_data()
    country_overviews, num_ethnicities_per_country, ethnicities_retained = _clean_and_prepare_data(data)

    _run_analyses(data)
    _export_tables(data, country_overviews, num_ethnicities_per_country, ethnicities_retained)
    _export_graphs(data)

    return {
        **data,
        "country_overviews": country_overviews,
        "num_ethnicities_per_country": num_ethnicities_per_country,
        "ethnicities_retained": ethnicities_retained,
    }
