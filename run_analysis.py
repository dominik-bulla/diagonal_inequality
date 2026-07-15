from __future__ import annotations

from diagonal_inequality.pipeline import run_pipeline

def main() -> None:
    results = run_pipeline()
    # Extract summary objects
    country_overviews = results["country_overviews"]

    # Extract country names (exclude summary keys)
    summary_keys = {
        "country_overviews",
        "num_ethnicities_per_country",
        "ethnicities_retained",
    }
    countries = [k for k in results.keys() if k not in summary_keys]
    print(f"Countries retained ({len(countries)}): {countries}")

if __name__ == "__main__":
    main()
