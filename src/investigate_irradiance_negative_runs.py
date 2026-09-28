from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

records = []

print("=" * 70)
print("DAYTIME NEGATIVE IRRADIANCE RUN ANALYSIS")
print("=" * 70)
print(f"Files: {len(files):,}")
print()

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(file)

        if "irradiance" not in df.columns:
            continue

        if "measured_on" not in df.columns:
            continue

        df["measured_on"] = pd.to_datetime(
            df["measured_on"],
            errors="coerce"
        )

        df["irradiance"] = pd.to_numeric(
            df["irradiance"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["measured_on", "irradiance"]
        ).sort_values("measured_on")

        # Daytime window
        # We deliberately use 08:00–16:00 first,
        # avoiding sunrise/sunset transition periods.
        daytime = df[
            (df["measured_on"].dt.hour >= 8) &
            (df["measured_on"].dt.hour <= 16)
        ].copy()

        if daytime.empty:
            continue

        # Focus on meaningful negative values
        daytime["negative_type"] = np.select(
            [
                (daytime["irradiance"] < 0) &
                (daytime["irradiance"] > -10),

                (daytime["irradiance"] <= -10) &
                (daytime["irradiance"] > -1000),

                daytime["irradiance"] <= -1000,
            ],
            [
                "small_negative",
                "medium_negative",
                "large_negative",
            ],
            default="not_negative"
        )

        neg = daytime[
            daytime["negative_type"] != "not_negative"
        ].copy()

        if neg.empty:
            continue

        # Basic information for this daily file
        records.append({
            "date": daytime["measured_on"].dt.date.iloc[0],
            "negative_count": len(neg),
            "small_negative": int(
                (neg["negative_type"] == "small_negative").sum()
            ),
            "medium_negative": int(
                (neg["negative_type"] == "medium_negative").sum()
            ),
            "large_negative": int(
                (neg["negative_type"] == "large_negative").sum()
            ),
            "min_irradiance": neg["irradiance"].min(),
            "max_irradiance": neg["irradiance"].max(),
        })

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

result = pd.DataFrame(records)

if not result.empty:

    result["date"] = pd.to_datetime(result["date"])

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month

    print()
    print("=" * 70)
    print("TOP DAYS WITH DAYTIME NEGATIVE IRRADIANCE")
    print("=" * 70)

    print(
        result.sort_values(
            "negative_count",
            ascending=False
        ).head(30).to_string(index=False)
    )

    print()
    print("=" * 70)
    print("MONTHLY SUMMARY")
    print("=" * 70)

    monthly = (
        result
        .groupby(["year", "month"])
        .agg(
            days_with_negative=("date", "count"),
            negative_count=("negative_count", "sum"),
            small_negative=("small_negative", "sum"),
            medium_negative=("medium_negative", "sum"),
            large_negative=("large_negative", "sum"),
        )
        .reset_index()
    )

    print(monthly.to_string(index=False))

    output_file = (
        OUTPUT_DIR /
        "irradiance_daytime_negative_runs.csv"
    )

    result.to_csv(output_file, index=False)

    monthly_file = (
        OUTPUT_DIR /
        "irradiance_daytime_negative_monthly.csv"
    )

    monthly.to_csv(monthly_file, index=False)

    print()
    print("Saved:")
    print(output_file)
    print(monthly_file)

else:
    print("No daytime negative irradiance found.")