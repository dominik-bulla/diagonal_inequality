from __future__ import annotations
from typing import Any
import pandas as pd

from .config import (
    OUT_DATAPOINTS,
    OUT_ETHNICITIES_RETAINED,
    OUT_NUM_ETHNICITIES_PER_COUNTRY,
    OUT_DISTRIBUTION_WEALTH,
    OUT_DISTRIBUTION_ACROSS_QUINTILES,
    OUT_MEAN_QUINTILES,
    OUT_PAIRWISE_RESULTS,
    PROCESSED_DIR,
    RESULTS_TABLE_DIR,
    OUT_ETHNIC_WEALTH_GAPS_MAX_TABLE,
    OUT_ETHNIC_WEALTH_GAPS_AVG_TABLE,
    OUT_ETHNIC_WEALTH_GAPS_GINI_TABLE
)


def ensure_directories() -> None:
    """
    Ensure that all required output directories exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def table_country_overviews(datapoints: pd.DataFrame) -> None:
    """
    Export global datapoints overview (all ethnicities before filtering).
    """
    datapoints.to_csv(OUT_DATAPOINTS, index=False, encoding="utf-8-sig")


def table_ethnicities_retained(ethnicities_retained: pd.DataFrame) -> None:
    """
    Export table of ethnic groups retained after filtering.
    """
    ethnicities_retained.to_csv(
        OUT_ETHNICITIES_RETAINED,
        index=False,
        encoding="utf-8-sig",
    )


def table_num_ethnicities_per_country(num_ethnicities_per_country: pd.DataFrame) -> None:
    """
    Export number of retained ethnic groups per country.
    """
    num_ethnicities_per_country.to_csv(
        OUT_NUM_ETHNICITIES_PER_COUNTRY,
        index=False,
        encoding="utf-8-sig",
    )


def _iterate_countries(data: dict[str, dict[str, Any]]):
    """
    Helper: iterate over country-level payloads, skipping global keys.
    """
    excluded_keys = {
        "datapoints",
        "num_ethnicities_per_country",
        "ethnicities_retained",
    }

    i = 1
    for country, payload in data.items():
        if country in excluded_keys:
            continue
        yield i, country, payload
        i += 1


def table_distribution_wealth(data: dict[str, dict[str, Any]]) -> None:
    """
    Export wealth distribution per country (full distribution table).
    """
    for i, country, payload in _iterate_countries(data):
        out_path = OUT_DISTRIBUTION_WEALTH.with_name(
            OUT_DISTRIBUTION_WEALTH.name.format(i=i, country=country)
        )

        pd.DataFrame(payload["distribution_wealth"]).to_csv(
            out_path,
            index=True,
            encoding="utf-8-sig",
        )


def table_distribution_across_quintiles(data: dict[str, dict[str, Any]]) -> None:
    """
    Export distribution of ethnic groups across wealth quintiles per country.
    """
    for i, country, payload in _iterate_countries(data):
        out_path = OUT_DISTRIBUTION_ACROSS_QUINTILES.with_name(
            OUT_DISTRIBUTION_ACROSS_QUINTILES.name.format(i=i, country=country)
        )

        pd.DataFrame(payload["distribution_across_quintiles"]).to_csv(
            out_path,
            index=True,
            encoding="utf-8-sig",
        )


def table_mean_quintiles(data: dict[str, dict[str, Any]]) -> None:
    """
    Export mean wealth rank (quintile) per ethnic group and country.
    """
    for i, country, payload in _iterate_countries(data):
        out_path = OUT_MEAN_QUINTILES.with_name(
            OUT_MEAN_QUINTILES.name.format(i=i, country=country)
        )

        pd.DataFrame(payload["mean_quintiles"]).to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
        )


def table_pairwise_results(data: dict[str, dict[str, Any]]) -> None:
    """
    Export pairwise statistical comparisons between ethnic groups per country.
    """
    for i, country, payload in _iterate_countries(data):
        out_path = OUT_PAIRWISE_RESULTS.with_name(
            OUT_PAIRWISE_RESULTS.name.format(i=i, country=country)
        )

        pd.DataFrame(payload["pairwise_results"]).to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
        )


def graphs_wealth_gaps_max_results(datapoints: pd.DataFrame) -> None:
    """
    Table the maximum absolute pairwise ethnic wealth gap for each country.
    """
    datapoints.to_csv(OUT_ETHNIC_WEALTH_GAPS_MAX_TABLE, index=False, encoding="utf-8-sig")


def graphs_wealth_gaps_avg_results(datapoints: pd.DataFrame) -> None:
    """
    Table the average absolute pairwise ethnic wealth gap for each country.
    """
    datapoints.to_csv(OUT_ETHNIC_WEALTH_GAPS_AVG_TABLE, index=False, encoding="utf-8-sig")


def graphs_wealth_gaps_gini_results(datapoints: pd.DataFrame) -> None:
    """
    Table the gini-based pairwise ethnic wealth gap for each country.
    """
    datapoints.to_csv(OUT_ETHNIC_WEALTH_GAPS_GINI_TABLE, index=False, encoding="utf-8-sig")








