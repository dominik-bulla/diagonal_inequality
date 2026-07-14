from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat

import re

from .config import DHS_DIR, DHS_VARS_PATH, MICS_DIR, MICS_COUNTRIES, DHS_COUNTRIES, WEALTH_INDEX_MAPPING, WEALTH_ORDER

def sanitize(name: str) -> str:
    name = str(name).strip().replace(" ", "_")
    return re.sub(r"[^\w_]", "", name) or "var"


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out





def normalize_wealth_index(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower().replace(WEALTH_INDEX_MAPPING)
    return pd.Categorical(normalized, categories=WEALTH_ORDER, ordered=True)


def map_value_labels(df: pd.DataFrame, meta: Any, column: str) -> pd.Series:
    try:
        label_set = meta.variable_to_label.get(column)
        value_map = meta.value_labels.get(label_set, {})
        if value_map:
            return df[column].map(value_map).fillna(df[column])
    except Exception:
        pass
    return df[column]


def find_design_var(
    df: pd.DataFrame,
    preferred_exact: list[str] | None = None,
    fallback_exact: list[str] | None = None,
    substrings: list[str] | None = None,
) -> str | None:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred_exact = preferred_exact or []
    fallback_exact = fallback_exact or []
    substrings = substrings or []

    for name in preferred_exact:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for name in fallback_exact:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    candidates = [c for c in cols if any(sub in c.lower() for sub in substrings)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return min(candidates, key=len)
    return None


def apply_mics_country_fixes(country: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if country in {"Republic of North Macedonia", "North Macedonia"} and "hhweight" in df.columns and "wmweight" not in df.columns:
        df = df.rename(columns={"hhweight": "wmweight"})
        notes.append("Renamed hhweight to wmweight.")
    return df, notes


def apply_dhs_country_fixes(country: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if country == "Uganda" and "wealth_index" in df.columns:
        df["wealth_index"] = normalize_wealth_index(df["wealth_index"])
        notes.append("Normalized Uganda wealth_index labels.")
    return df, notes




def list_dta_columns(path: Path) -> list[str]:
    _, meta = pyreadstat.read_dta(str(path), apply_value_formats=False, metadataonly=True)
    return list(meta.column_names)


def country_from_mics_filename(path: Path) -> str:
    stem = path.stem
    try:
        country, _suffix = stem.rsplit("_", 1)
    except ValueError:
        country = stem
    return country


def load_mics() -> dict[str, dict[str, Any]]:
    n_country = len(MICS_COUNTRIES["country"])
    n_year = len(MICS_COUNTRIES["year"])
    n_dataset = len(MICS_COUNTRIES["dataset"])
    if not (n_country == n_year == n_dataset):
        raise ValueError(
            "MICS_COUNTRIES in config.py must have equally long "
            "'country', 'year', and 'dataset' lists."
        )
    data_mics: dict[str, dict[str, Any]] = {}
    for country, year, dataset_name in zip(
        MICS_COUNTRIES["country"],
        MICS_COUNTRIES["year"],
        MICS_COUNTRIES["dataset"],
    ):
        fpath = MICS_DIR / dataset_name
        if not fpath.exists():
            continue
        try:
            df, meta = pyreadstat.read_sav(str(fpath), apply_value_formats=False)
        except Exception:
            continue
        for col in ["ethnicity", "windex5"]:
            if col in df.columns:
                df[col] = map_value_labels(df, meta, col)
        df, _ = apply_mics_country_fixes(country, df)
        psu_col = find_design_var(
            df,
            preferred_exact=["psu", "cluster", "hcluster", "v021"],
            fallback_exact=["HH1"],
            substrings=["psu", "cluster"],
        )
        stratum_col = find_design_var(
            df,
            preferred_exact=["stratum", "strata", "strat", "v022", "v023"],
            fallback_exact=["HH6"],
            substrings=["strat", "strata"],
        )
        keep_cols = ["wmweight", "ethnicity", "windex5", "wscore"]
        rename_map = {
            "wmweight": "sampling_weight_women",
            "ethnicity": "ethnicity",
            "windex5": "wealth_index",
            "wscore": "wealth_score",
        }
        if psu_col:
            keep_cols.append(psu_col)
            rename_map[psu_col] = "PSU"
        if stratum_col:
            keep_cols.append(stratum_col)
            rename_map[stratum_col] = "stratum"
        present_keep = [c for c in keep_cols if c in df.columns]
        df = df[present_keep].rename(columns=rename_map).copy()
        if "sampling_weight_women" in df.columns:
            df = df.loc[df["sampling_weight_women"] != 0].copy()
        if "wealth_index" in df.columns:
            df["wealth_index"] = normalize_wealth_index(df["wealth_index"])
        if "stratum" in df.columns:
            df["stratum"] = df["stratum"].astype("category")
        # Skip datasets missing required columns
        required = ["ethnicity", "wealth_score", "sampling_weight_women"]
        if not all(c in df.columns for c in required):
            continue
        data_mics[country] = {
            "source": "MICS",
            "year": year,
            "dataset": df,
            "meta": getattr(meta, "column_names_to_labels", {}),
        }
    return data_mics

def load_dhs() -> dict[str, dict[str, Any]]:
    vars_df = pd.read_csv(DHS_VARS_PATH, sep=None, engine="python")
    keep_mask = vars_df["keep"] == 1
    keep_vars = set(vars_df.loc[keep_mask, "original_name"].astype(str))
    keep_vars |= {"caseid", "v000", "v005"}
    rename_map_planned = {
        str(row["original_name"]): sanitize(row["new_label"])
        for _, row in vars_df.loc[keep_mask, ["original_name", "new_label"]]
        .fillna({"new_label": ""})
        .iterrows()
        if str(row["new_label"]).strip() != ""
    }
    n_country = len(DHS_COUNTRIES["country"])
    n_year = len(DHS_COUNTRIES["year"])
    n_dataset = len(DHS_COUNTRIES["dataset"])
    if not (n_country == n_year == n_dataset):
        raise ValueError(
            "DHS_COUNTRIES in config.py must have equally long "
            "'country', 'year', and 'dataset' lists."
        )
    data_dhs: dict[str, dict[str, Any]] = {}
    for country, year, dataset_name in zip(
        DHS_COUNTRIES["country"],
        DHS_COUNTRIES["year"],
        DHS_COUNTRIES["dataset"],
    ):
        fpath = DHS_DIR / dataset_name
        if not fpath.exists():
            continue
        try:
            available = set(list_dta_columns(fpath))
            use_cols = sorted(available & keep_vars)
            if not use_cols:
                continue
            df, meta = pyreadstat.read_dta(
                str(fpath),
                usecols=use_cols,
                apply_value_formats=True,
            )
        except Exception:
            continue
        rename_map_actual = {
            old: new for old, new in rename_map_planned.items() if old in df.columns
        }
        if rename_map_actual:
            df = df.rename(columns=rename_map_actual)
        if len(df.columns) != len(set(df.columns)):
            df.columns = make_unique(list(df.columns))
        keep = [
            "sample_weight",
            "ethnicity",
            "wealth_index",
            "wealth_index_factor",
            "primary_sampling_unit",
            "sample_strata_sampling",
        ]
        rename_map = {
            "sample_weight": "sampling_weight_women",
            "ethnicity": "ethnicity",
            "wealth_index": "wealth_index",
            "wealth_index_factor": "wealth_score",
            "primary_sampling_unit": "PSU",
            "sample_strata_sampling": "stratum",
        }
        present_keep = [c for c in keep if c in df.columns]
        df = df[present_keep].rename(columns=rename_map).copy()
        if "wealth_index" in df.columns:
            df["wealth_index"] = normalize_wealth_index(df["wealth_index"])
        if "stratum" in df.columns:
            df["stratum"] = df["stratum"].astype("category")
        df, _ = apply_dhs_country_fixes(country, df)
        # Skip datasets missing required columns
        required = ["ethnicity", "wealth_score", "sampling_weight_women"]
        if not all(c in df.columns for c in required):
            continue
        data_dhs[country] = {
            "source": "DHS",
            "year": year,
            "dataset": df,
            "meta": getattr(meta, "column_names_to_labels", {}),
        }
    return data_dhs
