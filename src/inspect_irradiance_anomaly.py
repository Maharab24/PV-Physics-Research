from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/processed/system33_daily")

TARGET_DATES = [
    "2017-06-10",
    "2017-06-11",
    "2017-06-12",
]

files = sorted(INPUT_DIR.rglob("*.parquet"))

print("=" * 80)
print("IRRADIANCE ANOMALY INSPECTION")
print("=" * 80)
print(f"Total files: {len(files):,}")
print()

for target_date in TARGET_DATES:

    print("=" * 80)
    print(f"SEARCHING: {target_date}")
    print("=" * 80)

    found = False

    for file in files:

        try:
            df = pd.read_parquet(
                file,
                columns=["measured_on", "irradiance"]
            )

        except Exception:
            continue

        df["measured_on"] = pd.to_datetime(
            df["measured_on"],
            errors="coerce"
        )

        df["irradiance"] = pd.to_numeric(
            df["irradiance"],
            errors="coerce"
        )

        target = df[
            df["measured_on"].dt.strftime("%Y-%m-%d")
            == target_date
        ].copy()

        if target.empty:
            continue

        found = True

        target = target.sort_values("measured_on")

        negative = target[
            (target["measured_on"].dt.hour >= 8) &
            (target["measured_on"].dt.hour <= 16) &
            (target["irradiance"] < 0)
        ].copy()

        print(f"File: {file}")
        print(f"Total rows: {len(target):,}")
        print(f"Daytime negative rows: {len(negative):,}")
        print()

        if negative.empty:
            print("No daytime negative irradiance.")
            print()
            break

        print("First 30 negative observations:")
        print(
            negative[
                ["measured_on", "irradiance"]
            ].head(30).to_string(index=False)
        )

        print()
        print("Last 30 negative observations:")
        print(
            negative[
                ["measured_on", "irradiance"]
            ].tail(30).to_string(index=False)
        )

        print()
        print("Statistics:")
        print(
            negative["irradiance"].describe().to_string()
        )

        print()
        print("Most frequent negative values:")
        print(
            negative["irradiance"]
            .value_counts()
            .head(20)
            .to_string()
        )

        print()
        break

    if not found:
        print(f"Could not find data for {target_date}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)