from __future__ import annotations

"""Data-cleaning utilities for wealth-by-ethnicity analyses.

This module contains the preprocessing steps applied to the country-level
DHS and MICS datasets before analysis. All public functions are designed to
operate in-place on the shared ``data`` dictionary used throughout the
project and are called from ``pipeline.py``.

Expected structure of ``data``
------------------------------
Each key is a country name and each value is a dictionary containing at
least a pandas DataFrame stored under ``"dataset"``. Additional metadata,
such as the survey source, may also be present.
"""

from typing import Any, Iterable
import re
import unicodedata

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .config import CANONICAL_MAP, DROP_EXACT_NORM, MIN_ETHNICITY_N, WEALTH_ORDER

DataDict = dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Text normalization and validation helpers
# ---------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    """Return a normalized string for text matching.

    The function removes accents, standardizes apostrophes, lowercases text,
    and collapses repeated whitespace. Missing values are converted to an
    empty string.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("’", "'").replace("´", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_canonical_ethnicity(raw: Any) -> Any:
    """Map a raw ethnicity label to its canonical form.

    Labels flagged for removal are converted to ``pd.NA``. If a label is not
    found in the canonical mapping, a cleaned title-cased version is returned
    as a fallback.
    """
    norm = normalize_text(raw)

    if norm in DROP_EXACT_NORM:
        return pd.NA
    if norm in CANONICAL_MAP:
        return CANONICAL_MAP[norm]

    compact = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    if compact in CANONICAL_MAP:
        return CANONICAL_MAP[compact]

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.NA

    text = str(raw).strip()
    return text.title() if text else pd.NA


def validate_required_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    country: str,
    source: str,
) -> None:
    """Raise an error if a required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source}/{country}: missing required columns {missing}")


# ---------------------------------------------------------------------------
# Internal dataset helpers
# ---------------------------------------------------------------------------
def _get_dataset_copy(info: dict[str, Any]) -> pd.DataFrame:
    """Return a defensive copy of a country dataset."""
    return info["dataset"].copy()


def _store_dataset(info: dict[str, Any], df: pd.DataFrame) -> None:
    """Write a cleaned dataset back into a country record."""
    info["dataset"] = df


def _empty_ethnicity_row(country: str) -> dict[str, Any]:
    """Return a placeholder row for countries without usable ethnicity data."""
    return {"country": country, "ethnicity": None, "count_in_country": 0}


# ---------------------------------------------------------------------------
# Public cleaning functions used by pipeline.py
# ---------------------------------------------------------------------------
def drop_missing_wealth_index(data: DataDict) -> None:
    """Remove rows with missing wealth-index information from each dataset."""
    for info in data.values():
        df = _get_dataset_copy(info)
        df = df.dropna(subset=["wealth_index"]).reset_index(drop=True)
        _store_dataset(info, df)



def clean_ethnicity_labels(data: DataDict) -> None:
    """Standardize ethnicity labels and drop rows with unresolved values."""
    for info in data.values():
        df = _get_dataset_copy(info)
        df["ethnicity"] = df["ethnicity"].map(to_canonical_ethnicity)
        df = df.dropna(subset=["ethnicity"]).reset_index(drop=True)
        _store_dataset(info, df)



def export_all_ethnicities(data: DataDict) -> pd.DataFrame:
    """Return a country-by-ethnicity count table before size filtering."""
    rows: list[dict[str, Any]] = []

    for country, info in data.items():
        ethnicity_values = info["dataset"].get("ethnicity")
        if ethnicity_values is None:
            rows.append(_empty_ethnicity_row(country))
            continue

        counts = pd.Series(ethnicity_values).dropna().value_counts()
        if counts.empty:
            rows.append(_empty_ethnicity_row(country))
            continue

        for ethnicity, count in counts.items():
            rows.append(
                {
                    "country": country,
                    "ethnicity": ethnicity,
                    "count_in_country": int(count),
                }
            )

    return pd.DataFrame(rows)



def retain_large_ethnic_groups(
    data: DataDict,
    min_n: int = MIN_ETHNICITY_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only ethnic groups with at least ``min_n`` observations.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A pair containing:
        1. a country-level summary of observed and retained ethnicities, and
        2. a detailed table of retained ethnic groups and their unweighted
           counts.
    """
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    allowed_ethnicities: dict[str, set[str]] = {}
    for country, info in data.items():
        df = info["dataset"]
        if "ethnicity" not in df.columns:
            continue
        counts = df["ethnicity"].value_counts(dropna=True)
        retained = counts[counts >= min_n].index.tolist()
        allowed_ethnicities[country] = set(retained)
        summary_rows.append(
            {
                "dataset": country,
                "n_ethnicities": int(counts.shape[0]),
                f"n_eth_ge_{min_n}": int((counts >= min_n).sum()),
            }
        )
        for ethnicity in retained:
            detail_rows.append(
                {
                    "dataset": country,
                    "ethnicity": ethnicity,
                    "n_unweighted": int(counts[ethnicity]),
                }
            )
    for country, info in data.items():
        df = _get_dataset_copy(info)
        if "ethnicity" in df.columns:
            keep = allowed_ethnicities.get(country, set())
            df = df[df["ethnicity"].isin(keep)].copy()

            # Remove category levels that are no longer represented.
            if isinstance(df["ethnicity"].dtype, CategoricalDtype):
                df["ethnicity"] = df["ethnicity"].cat.remove_unused_categories()

        _store_dataset(info, df)
    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    detail_df = (
        pd.DataFrame(detail_rows)
        .sort_values(["dataset", "n_unweighted"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary_df, detail_df

def add_dummies_and_scale_weights(data: DataDict) -> None:
    """Scale DHS weights and append dummy variables for analysis."""
    for info in data.values():
        df = _get_dataset_copy(info)

        if info.get("source") == "DHS" and "sampling_weight_women" in df.columns:
            # DHS weights are commonly stored as six-digit integers.
            df["sampling_weight_women"] = df["sampling_weight_women"] / 1_000_000

        if "ethnicity" in df.columns:
            ethnicity_dummies = pd.get_dummies(
                df["ethnicity"],
                prefix="ethnicity",
                dtype=int,
                drop_first=False,
            )
            df = pd.concat([df, ethnicity_dummies], axis=1)

        if "wealth_index" in df.columns:
            wealth_dummies = pd.get_dummies(
                df["wealth_index"],
                prefix="wealth",
                dtype=int,
                drop_first=False,
            )
            df = pd.concat([df, wealth_dummies], axis=1)

        _store_dataset(info, df)



def validate_final_data(data: DataDict) -> None:
    """Run final consistency checks before analysis starts."""
    required_columns = {"ethnicity", "wealth_score", "sampling_weight_women"}

    for country, info in data.items():
        df = info["dataset"]
        validate_required_columns(df, required_columns, country, info["source"])

        if "sampling_weight_women" in df.columns and (df["sampling_weight_women"] == 0).any():
            raise ValueError(f"{country}: zero sampling weights remain.")

        if "wealth_index" in df.columns:
            observed = set(df["wealth_index"].dropna().astype(str).unique())
            unexpected = observed - set(WEALTH_ORDER)
            if unexpected:
                raise ValueError(
                    f"{country}: unexpected wealth_index labels: {sorted(unexpected)}"
                )

        if "ethnicity" in df.columns and df["ethnicity"].isna().any():
            raise ValueError(f"{country}: missing ethnicity values remain after cleaning.")
