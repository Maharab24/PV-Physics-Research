from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

bins = {
    "(-1, 0)": 0,
    "(-10, -1]": 0,
    "(-50, -10]": 0,
    "(-100, -50]": 0,
    "(-500, -100]": 0,
    "(-1000, -500]": 0,
    "<= -1000": 0,
}

total_negative = 0
total_valid = 0

print("=" * 70)
print("NEGATIVE IRRADIANCE MAGNITUDE ANALYSIS")
print("=" * 70)
print(f"Files: {len(files):,}")
print()

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(
            file,
            columns=["irradiance"]
        )

        if "irradiance" not in df.columns:
            continue

        s = pd.to_numeric(
            df["irradiance"],
            errors="coerce"
        ).dropna()

        negative = s[s < 0]

        total_negative += len(negative)
        total_valid += len(s)

        bins["(-1, 0)"] += int(
            ((negative > -1) & (negative < 0)).sum()
        )

        bins["(-10, -1]"] += int(
            ((negative > -10) & (negative <= -1)).sum()
        )

        bins["(-50, -10]"] += int(
            ((negative > -50) & (negative <= -10)).sum()
        )

        bins["(-100, -50]"] += int(
            ((negative > -100) & (negative <= -50)).sum()
        )

        bins["(-500, -100]"] += int(
            ((negative > -500) & (negative <= -100)).sum()
        )

        bins["(-1000, -500]"] += int(
            ((negative > -1000) & (negative <= -500)).sum()
        )

        bins["<= -1000"] += int(
            (negative <= -1000).sum()
        )

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

rows = []

for category, count in bins.items():

    percentage = (
        count / total_negative * 100
        if total_negative > 0
        else 0
    )

    rows.append({
        "range": category,
        "count": count,
        "percentage_of_negative": percentage,
    })

result = pd.DataFrame(rows)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(result.to_string(index=False))

print()
print(f"Total irradiance observations: {total_valid:,}")
print(f"Total negative irradiance:    {total_negative:,}")

output_file = (
    OUTPUT_DIR /
    "irradiance_negative_magnitude.csv"
)

result.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)