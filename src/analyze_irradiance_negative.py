from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

hour_stats = {
    hour: {
        "total": 0,
        "negative": 0,
        "zero": 0,
        "positive": 0,
    }
    for hour in range(24)
}

monthly_stats = {}

print("=" * 70)
print("IRRADIANCE NEGATIVE-VALUE TEMPORAL ANALYSIS")
print("=" * 70)
print(f"Files: {len(files):,}")
print()

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(
            file,
            columns=["measured_on", "irradiance"]
        )

        df["measured_on"] = pd.to_datetime(
            df["measured_on"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["measured_on", "irradiance"]
        )

        df["hour"] = df["measured_on"].dt.hour
        df["year_month"] = (
            df["measured_on"]
            .dt.to_period("M")
            .astype(str)
        )

        # --------------------------------------------------
        # Hourly analysis
        # --------------------------------------------------

        for hour, group in df.groupby("hour"):

            values = group["irradiance"]

            hour_stats[hour]["total"] += len(values)
            hour_stats[hour]["negative"] += int(
                (values < 0).sum()
            )
            hour_stats[hour]["zero"] += int(
                (values == 0).sum()
            )
            hour_stats[hour]["positive"] += int(
                (values > 0).sum()
            )

        # --------------------------------------------------
        # Monthly analysis
        # --------------------------------------------------

        for month, group in df.groupby("year_month"):

            values = group["irradiance"]

            if month not in monthly_stats:
                monthly_stats[month] = {
                    "total": 0,
                    "negative": 0,
                    "zero": 0,
                    "positive": 0,
                }

            monthly_stats[month]["total"] += len(values)
            monthly_stats[month]["negative"] += int(
                (values < 0).sum()
            )
            monthly_stats[month]["zero"] += int(
                (values == 0).sum()
            )
            monthly_stats[month]["positive"] += int(
                (values > 0).sum()
            )

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

# ----------------------------------------------------------
# Hourly output
# ----------------------------------------------------------

hour_rows = []

for hour in range(24):

    d = hour_stats[hour]

    negative_pct = (
        d["negative"] / d["total"] * 100
        if d["total"] > 0
        else 0
    )

    hour_rows.append({
        "hour": hour,
        "total": d["total"],
        "negative": d["negative"],
        "negative_percent": negative_pct,
        "zero": d["zero"],
        "positive": d["positive"],
    })

hour_df = pd.DataFrame(hour_rows)

# ----------------------------------------------------------
# Monthly output
# ----------------------------------------------------------

month_rows = []

for month in sorted(monthly_stats):

    d = monthly_stats[month]

    negative_pct = (
        d["negative"] / d["total"] * 100
        if d["total"] > 0
        else 0
    )

    month_rows.append({
        "year_month": month,
        "total": d["total"],
        "negative": d["negative"],
        "negative_percent": negative_pct,
        "zero": d["zero"],
        "positive": d["positive"],
    })

month_df = pd.DataFrame(month_rows)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

hour_file = OUTPUT_DIR / "irradiance_negative_by_hour.csv"
month_file = OUTPUT_DIR / "irradiance_negative_by_month.csv"

hour_df.to_csv(hour_file, index=False)
month_df.to_csv(month_file, index=False)

print()
print("=" * 70)
print("NEGATIVE IRRADIANCE BY HOUR")
print("=" * 70)

print(
    hour_df.to_string(index=False)
)

print()
print("=" * 70)
print("MONTHLY SUMMARY")
print("=" * 70)

print(
    month_df.to_string(index=False)
)

print()
print("Saved:")
print(hour_file)
print(month_file)