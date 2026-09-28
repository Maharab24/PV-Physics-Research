from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

hourly = {
    h: {
        "total": 0,
        "negative": 0,
        "small_negative": 0,   # -10 < x < 0
        "medium_negative": 0,  # -1000 < x <= -10
        "large_negative": 0,   # <= -1000
        "zero": 0,
        "positive": 0,
    }
    for h in range(24)
}

missing_irradiance_files = []

print("=" * 70)
print("NEGATIVE IRRADIANCE TIMING ANALYSIS")
print("=" * 70)
print(f"Files: {len(files):,}")
print()

for i, file in enumerate(files, 1):

    try:
        # Daily files are small, so reading the whole file is safe
        df = pd.read_parquet(file)

        if "irradiance" not in df.columns:
            missing_irradiance_files.append(str(file))
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

        df = df.dropna(subset=["measured_on", "irradiance"])

        df["hour"] = df["measured_on"].dt.hour

        for hour, group in df.groupby("hour"):

            x = group["irradiance"]

            hourly[hour]["total"] += len(x)

            hourly[hour]["negative"] += int((x < 0).sum())

            hourly[hour]["small_negative"] += int(
                ((x >= -10) & (x < 0)).sum()
            )

            hourly[hour]["medium_negative"] += int(
                ((x >= -1000) & (x < -10)).sum()
            )

            hourly[hour]["large_negative"] += int(
                (x < -1000).sum()
            )

            hourly[hour]["zero"] += int((x == 0).sum())

            hourly[hour]["positive"] += int((x > 0).sum())

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

rows = []

for hour in range(24):

    d = hourly[hour]

    total = d["total"]
    negative = d["negative"]

    rows.append({
        "hour": hour,
        "total": total,
        "negative": negative,
        "negative_percent": (
            negative / total * 100
            if total > 0 else 0
        ),
        "small_negative": d["small_negative"],
        "small_negative_percent": (
            d["small_negative"] / total * 100
            if total > 0 else 0
        ),
        "medium_negative": d["medium_negative"],
        "medium_negative_percent": (
            d["medium_negative"] / total * 100
            if total > 0 else 0
        ),
        "large_negative": d["large_negative"],
        "large_negative_percent": (
            d["large_negative"] / total * 100
            if total > 0 else 0
        ),
        "zero": d["zero"],
        "positive": d["positive"],
    })

result = pd.DataFrame(rows)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    result[
        [
            "hour",
            "total",
            "negative",
            "negative_percent",
            "small_negative",
            "medium_negative",
            "large_negative",
            "zero",
            "positive",
        ]
    ].to_string(index=False)
)

print()
print(f"Files missing irradiance: {len(missing_irradiance_files)}")

output = OUTPUT_DIR / "irradiance_negative_timing.csv"
result.to_csv(output, index=False)

print()
print("Saved:")
print(output)