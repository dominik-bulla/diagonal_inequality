# Chapter 11: Diagonal inequality? The global prevalence of the racial wealth divide and what this means for policy

This repository contains the **analysis code accompanying the publication**:

> **Lenhardt, Shaheen, Bulla (2026)**
> *Chapter 11: Diagonal inequality? The global prevalence of the racial wealth divide and what this means for policy*

This repository provides the full analysis pipeline used to generate the results reported in the publication. It processes harmonized survey microdata from:

* Demographic and Health Surveys (DHS)
* Multiple Indicator Cluster Surveys (MICS)

to quantify ethnic disparities in wealth distributions across countries.

---

## Overview

The project implements a fully reproducible pipeline that:

1. Loads DHS and MICS microdata
2. Cleans and harmonizes ethnicity and wealth indicators
3. Computes weighted wealth distributions and ethnic inequalities
4. Performs statistical comparisons between ethnic groups
5. Exports results as tables and publication-ready figures

---

## Project Structure

```text
.
├── data/         # must create
│   ├── DHS/      # must create
│   ├── MICS/     # must create
│   └── processed/
├── results/
│   ├── tables/
│   └── graphs/
├── src/
│   └── diagonal_inequality/
├── run_analysis.py
├── requirements.txt
└── README.md
```

---

## Important Note

Reproducibility depends on:

* Correct dataset versions (see below)
* Consistent file naming
* Alignment with the configuration in `config.py`

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Recreate the Environment from Scratch

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Running the Analysis

```bash
python run_analysis.py
```

---

## Outputs

Results are written to the `results/` directory.

### Tables (CSV)

* Country summaries
* Wealth distributions
* Ethnicity comparisons

### Figures (PDF, PNG, TIFF)

* Wealth gap figures
* Heatmaps
* Spine plots

---

## Data Availability

This project uses microdata from:

* Demographic and Health Surveys (DHS)
* Multiple Indicator Cluster Surveys (MICS)

Due to licensing restrictions:

* Raw data **cannot be redistributed**.
* This repository therefore provides **code only**.

To reproduce the analysis, the data must be requested and downloaded from:

* DHS: https://dhsprogram.com/
* MICS: https://mics.unicef.org/

---

## Data Placement

Downloaded datasets should be stored in the following directory structure:

```text
data/
├── DHS/      # DHS datasets (.dta files)
└── MICS/     # MICS datasets (.sav files)
```

* DHS data should be provided as `.dta` files.
* MICS data should be provided as `.sav` files.

---

## Configuration

To ensure that datasets are loaded correctly, the country-specific file names must be registered in `config.py`.

Specifically, update the dictionaries:

* `DHS_COUNTRIES`
* `MICS_COUNTRIES`

These dictionaries define:

* the countries included in the analysis,
* the survey years,
* the corresponding dataset file names.

The file names specified there must match the files placed in the `data/DHS/` and `data/MICS/` directories.

---

## License

This repository contains analysis code only.

Use of DHS and MICS data remains subject to their respective licensing agreements.
