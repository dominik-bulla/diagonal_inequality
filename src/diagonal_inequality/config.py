"""Project-wide configuration for the wealth/ethnicity analysis pipeline.

This module centralizes:
- filesystem paths
- output file naming conventions
- survey metadata
- analysis thresholds and constants
- ethnicity/wealth normalization maps

The structure is intentionally conservative so existing analysis scripts can keep
importing the same constant names while benefiting from cleaner organization and
better documentation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

RUN_DATE = "20260401"


def _env_path(var_name: str, default: Path | str) -> Path:
    """Return a Path from an environment variable, falling back to `default`."""
    return Path(os.getenv(var_name, str(default))).resolve()


def _normalize_key(value: str) -> str:
    """Normalize free-text labels for robust dictionary lookup."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = _env_path("WEALTH_ETHNICITY_PROJECT_ROOT", ".")

DATA_DIR = _env_path("WEALTH_ETHNICITY_DATA_DIR", PROJECT_ROOT / "data")
MICS_DIR = _env_path("WEALTH_ETHNICITY_MICS_DIR", PROJECT_ROOT / "data" / "MICS")
DHS_DIR = _env_path("WEALTH_ETHNICITY_DHS_DIR", PROJECT_ROOT / "data" / "DHS")
DHS_VARS_PATH = _env_path(
    "WEALTH_ETHNICITY_DHS_VARS", DHS_DIR / "DHS_variables.csv"
)
PROCESSED_DIR = _env_path(
    "WEALTH_ETHNICITY_PROCESSED_DIR", DATA_DIR / "processed"
)
RESULTS_TABLE_DIR = _env_path(
    "WEALTH_ETHNICITY_RESULTS_TABLE_DIR", PROJECT_ROOT / "results" / "tables"
)
RESULTS_GRAPHS_DIR = _env_path(
    # Keep backward compatibility with the mistaken env var if it is still used.
    "WEALTH_ETHNICITY_RESULTS_GRAPHS_DIR",
    os.getenv(
        "WEALTH_ETHNICITY_RESULTS_TABLE_DIR",
        str(PROJECT_ROOT / "results" / "graphs"),
    ),
)


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

# Pickled outputs (kept here for convenience if re-enabled later)
# OUT_DATA_PICKLE = PROCESSED_DIR / f"data_{RUN_DATE}.pkl"
# OUT_RESULTS_PICKLE = PROCESSED_DIR / f"analysis_results_{RUN_DATE}.pkl"

OUT_DATAPOINTS = RESULTS_TABLE_DIR / f"00_country_overviews_{RUN_DATE}.csv"
OUT_NUM_ETHNICITIES_PER_COUNTRY = (
    RESULTS_TABLE_DIR / f"01_num_ethnicities_per_country_{RUN_DATE}.csv"
)
OUT_ETHNICITIES_RETAINED = (
    RESULTS_TABLE_DIR / f"02_ethnicities_retained_in_countries_{RUN_DATE}.csv"
)
OUT_DISTRIBUTION_WEALTH = RESULTS_TABLE_DIR / "03.{i}_wealth_in_{country}.csv"
OUT_DISTRIBUTION_ACROSS_QUINTILES = (
    RESULTS_TABLE_DIR / "04.{i}_wealth_across_ethnicities_in_{country}.csv"
)
OUT_MEAN_QUINTILES = RESULTS_TABLE_DIR / "05.{i}_mean_quintile_rank_{country}.csv"
OUT_PAIRWISE_RESULTS = RESULTS_TABLE_DIR / "06.{i}_pairwise_tests_{country}.csv"

OUT_ETHNIC_WEALTH_GAPS_MAX_PDF = RESULTS_GRAPHS_DIR / "01.1_ethnic_wealth_gaps_(max).pdf"
OUT_ETHNIC_WEALTH_GAPS_MAX_TIFF = RESULTS_GRAPHS_DIR / "01.1_ethnic_wealth_gaps_(max).tiff"
OUT_ETHNIC_WEALTH_GAPS_MAX_PNG = RESULTS_GRAPHS_DIR / "01.1_ethnic_wealth_gaps_(max).png"

OUT_ETHNIC_WEALTH_GAPS_AVG_PDF = RESULTS_GRAPHS_DIR / "01.2_ethnic_wealth_gaps_(avg).pdf"
OUT_ETHNIC_WEALTH_GAPS_AVG_TIFF = RESULTS_GRAPHS_DIR / "01.2_ethnic_wealth_gaps_(avg).tiff"
OUT_ETHNIC_WEALTH_GAPS_AVG_PNG = RESULTS_GRAPHS_DIR / "01.2_ethnic_wealth_gaps_(avg).png"

OUT_ETHNIC_WEALTH_GAPS_GINI_PDF = RESULTS_GRAPHS_DIR / "01.3_ethnic_wealth_gaps_(gini).pdf"
OUT_ETHNIC_WEALTH_GAPS_GINI_TIFF = RESULTS_GRAPHS_DIR / "01.3_ethnic_wealth_gaps_(gini).tiff"
OUT_ETHNIC_WEALTH_GAPS_GINI_PNG = RESULTS_GRAPHS_DIR / "01.3_ethnic_wealth_gaps_(gini).png"

OUT_SPINE_PLOT_PDF = RESULTS_GRAPHS_DIR / "02.{i}_spine_plot_{country}.pdf"
OUT_SPINE_PLOT_TIFF = RESULTS_GRAPHS_DIR / "02.{i}_spine_plot_{country}.tiff"
OUT_SPINE_PLOT_PNG = RESULTS_GRAPHS_DIR / "02.{i}_spine_plot_{country}.png"

OUT_HEATMAP_PDF = RESULTS_GRAPHS_DIR / "03.{i}_heatmap_{country}.pdf"
OUT_HEATMAP_TIFF = RESULTS_GRAPHS_DIR / "03.{i}_heatmap_{country}.tiff"
OUT_HEATMAP_PNG = RESULTS_GRAPHS_DIR / "03.{i}_heatmap_{country}.png"

OUT_RELATIVE_REPRESENTATION_POOREST_PDF = (
    RESULTS_GRAPHS_DIR / "04.1_relative_representation_poorest.pdf"
)
OUT_RELATIVE_REPRESENTATION_POOREST_TIFF = (
    RESULTS_GRAPHS_DIR / "04.1_relative_representation_poorest.tiff"
)
OUT_RELATIVE_REPRESENTATION_POOREST_PNG = (
    RESULTS_GRAPHS_DIR / "04.1_relative_representation_poorest.png"
)

OUT_RELATIVE_REPRESENTATION_RICHEST_PDF = (
    RESULTS_GRAPHS_DIR / "04.2_relative_representation_richest.pdf"
)
OUT_RELATIVE_REPRESENTATION_RICHEST_TIFF = (
    RESULTS_GRAPHS_DIR / "04.2_relative_representation_richest.tiff"
)
OUT_RELATIVE_REPRESENTATION_RICHEST_PNG = (
    RESULTS_GRAPHS_DIR / "04.2_relative_representation_richest.png"
)

# -----------------------------------------------------------------------------
# Analysis constants
# -----------------------------------------------------------------------------

WEALTH_ORDER = ["poorest", "poorer", "middle", "richer", "richest"]
QCOLS = [
    "wealth_poorest",
    "wealth_poorer",
    "wealth_middle",
    "wealth_richer",
    "wealth_richest",
]
QRANK = np.array([1, 2, 3, 4, 5], dtype=float)
MIN_ETHNICITY_N = 100
ALPHA = 0.05


# -----------------------------------------------------------------------------
# Survey metadata
# -----------------------------------------------------------------------------

MICS_COUNTRIES = {
    "country": [
        "Belize",
        "Central African Republic",
        "Chad",
        "Congo",
        "Costa Rica",
        "Cuba",
        "Dominican Rep",
        "DRC",
        "Georgia",
        "Guinea-Bissau",
        "Guyana",
        "Honduras",
        "Kazakhstan",
        "Kyrgyzstan",
        "Lao PDR",
        "Lesotho",
        "Montenegro",
        "Paraguay",
        "North Macedonia",
        "Serbia",
        "Suriname",
    ],
    "year": [
        "2015-16",
        "2018-19",
        "2019",
        "2014-15",
        "2018",
        "2019",
        "2019",
        "2017-18",
        "2018",
        "2018-19",
        "2019-20",
        "2019",
        "2015",
        "2018",
        "2017",
        "2018",
        "2018",
        "2016",
        "2018-19",
        "2019",
        "2018",
    ],
    "dataset": [
        "Belize_MICS5.sav",
        "Central African Rep_MICS6.sav",
        "Chad_MICS 6.sav",
        "Congo_MICS5.sav",
        "Costa Rica_MICS6.sav",
        "Cuba_MICS6.sav",
        "Dominican Rep_MICS6.sav",
        "DRC_MICS6.sav",
        "Georgia_MICS6.sav",
        "Guinea Bissau_MICS6.sav",
        "Guyana_MICS6.sav",
        "Honduras_MICS6.sav",
        "Kazakhstan_MICS5.sav",
        "Kyrgyzstan_MICS6.sav",
        "Lao_MICS6.sav",
        "Lesotho_MICS6.sav",
        "Montenegro_MICS6.sav",
        "Paraguary_MICS5.sav",
        "Republic of North Macedonia_MICS6.sav",
        "Serbia_MICS6.sav",
        "Suriname_MICS6.sav",
    ],
}

DHS_COUNTRIES = {
    "country": [
        "Afghanistan",
        "Albania",
        "Burkina Faso",
        "Benin",
        "Côte d’Ivoire",
        "Colombia",
        "Ethiopia",
        "Gabon",
        "Ghana",
        "Gambia",
        "Guinea",
        "Guatemala",
        "Jordan",
        "Kenya",
        "Mali",
        "Malawi",
        "Nigeria",
        "Nepal",
        "Philippines",
        "Sierra Leone",
        "Senegal",
        "Togo",
        "Uganda",
        "South Africa",
    ],
    "year": [
        "2015",
        "2017",
        "2021",
        "2017/18",
        "2021",
        "2015",
        "2016",
        "2019/2021",
        "2022",
        "2019/20",
        "2021",
        "2014/15",
        "2017/18",
        "2022",
        "2021",
        "2017",
        "2021",
        "2022",
        "2022",
        "2019",
        "2023",
        "2018",
        "2016",
        "2016",
    ],
    "dataset": [
        "AFIR71FL.DTA",
        "ALIR71FL.DTA",
        "BFIR81FL.DTA",
        "BJIR71FL.DTA",
        "CIIR81FL.DTA",
        "COIR72FL.DTA",
        "ETIR71FL.DTA",
        "GAIR71FL.DTA",
        "GHIR8CFL.DTA",
        "GMIR81FL.DTA",
        "GNIR82FL.DTA",
        "GUIR71FL.DTA",
        "JOIR74FL.DTA",
        "KEIR8CFL.DTA",
        "MLIR83FL.DTA",
        "MWIR7IFL.DTA",
        "NGIR81FL.DTA",
        "NPIR82FL.DTA",
        "PHIR82FL.DTA",
        "SLIR7AFL.DTA",
        "SNIR8SFL.DTA",
        "TGIR71FL.DTA",
        "UGIR7BFL.DTA",
        "ZAIR71FL.DTA",
    ],
}


# -----------------------------------------------------------------------------
# Label harmonization
# -----------------------------------------------------------------------------

WEALTH_INDEX_MAPPING = {
    "poorest": "poorest",
    "lowest": "poorest",
    "second": "poorer",
    "poorer": "poorer",
    "middle": "middle",
    "fourth": "richer",
    "richer": "richer",
    "richest": "richest",
    "highest": "richest",
    "le plus pauvre": "poorest",
    "pauvre": "poorer",
    "moyen": "middle",
    "quatrième": "richer",
    "riche": "richer",
    "le plus riche": "richest",
    "mais pobre": "poorest",
    "segundo": "poorer",
    "médio": "middle",
    "quarto": "richer",
    "mais rico": "richest",
    "más pobre": "poorest",
    "medio": "middle",
    "cuarto": "richer",
    "más rico": "richest",
    "lowest ": "poorest",
}

DROP_EXACT_NORM = {
    "autre ethnie",
    "autres",
    "autres ethnies",
    "dk",
    "dk, unsure",
    "dont know",
    "don't know",
    "hablante de otro idioma",
    "missing",
    "missing/dk",
    "mulato, mestizo / otro",
    "nan",
    "ne sait pas",
    "ninguna",
    "ninguno",
    "no responde",
    "no sabe",
    "no response",
    "none",
    "not specified",
    "not stated",
    "ns nr",
    "other",
    "other arab nationalities",
    "other beninois",
    "other burkinabe",
    "other/dk/missing",
    "other ethnicity",
    "other ethnicity/dk/missing",
    "other ethnic group",
    "other ethnic groups",
    "other malian",
    "other nationalities",
    "other sierra leone",
    "other terai caste",
    "other togolese",
    "other/does not want to declare",
    "others",
    "otra etnia",
    "outra etnia",
    "refused",
    "unknown",
    "unsure",
}

CANONICAL_MAP = {
    "haoussa": "Hausa",
    "hausa": "Hausa",
    "peul": "Fulani",
    "peul/foulbe": "Fulani",
    "peul foulbe": "Fulani",
    "fulbe": "Fulani",
    "fulani": "Fulani",
    "foulbe": "Fulani",
    "pulaar": "Fulani",
    "pashtun": "Pashtun",
    "tajik": "Tajik",
    "hazara": "Hazara",
    "uzbek": "Uzbek",
    "turkmen": "Turkmen",
    "baloch": "Baloch",
    "pashai": "Pashai",
    "nuristani": "Nuristani",
    "malinke": "Malinké",
    "mandingue": "Mandinka",
    "dioula": "Dioula",
    "baoule": "Baoulé",
    "bete": "Bété",
    "agni": "Agni",
    "abron": "Abron",
    "serbian": "Serb",
    "serb": "Serb",
    "macedonian": "Macedonian",
    "affar": "Afar",
    "hablante solo castellano": "Spanish speaker",
    "hablante solo guarani": "Guaraní speaker",
    "hablante guarani y castellano": "Bilingual (Guaraní/Spanish)",
    "amerindian": "Amerindian",
    "indigena": "Indigenous",
    "indigenous": "Indigenous",
    "mestizo": "Mestizo",
    "blanco": "White",
    "african/black": "Black",
    "arabe": "Arab",
    "armenian": "Armenian",
    "azeri": "Azeri",
    "azer i": "Azeri",
}

# Add normalized aliases so callers can use raw or cleaned keys interchangeably.
CANONICAL_MAP.update({_normalize_key(key): value for key, value in list(CANONICAL_MAP.items())})


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def render_path(template: Path, **kwargs: str | int) -> Path:
    """Render a parameterized Path template such as '{country}' or '{i}'."""
    return Path(str(template).format(**kwargs))


def ensure_output_dirs() -> None:
    """Create standard output directories if they do not already exist."""
    for path in (DATA_DIR, PROCESSED_DIR, RESULTS_TABLE_DIR, RESULTS_GRAPHS_DIR):
        path.mkdir(parents=True, exist_ok=True)
