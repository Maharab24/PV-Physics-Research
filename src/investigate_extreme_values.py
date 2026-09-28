from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "irradiance",
    "dc_power",
    "ac_power",
    "dc_voltage",
    "dc_current",
    "ac_voltage",
    "ac_current",
    "ambient_temp",
    "module_temp_1",
    "module_temp_2",
    "module_temp_3",
    "inverter_temp",
    "panel_temp",
    "battery_voltage",
    "power_factor",
]

# Store exact extreme values for each metric
extreme_values = {
    metric: Counter()
    for metric in METRICS
}

# Thresholds
LOW_THRESHOLD = -1000
HIGH_THRESHOLD = 3000

files = sorted(INPUT_DIR.rglob("*.parquet"))

print("=" * 70)
print("INVESTIGATING EXACT EXTREME VALUES")
print("=" * 70)
print(f"Files: {len(files):,}")
print()

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(file)

        for metric in METRICS:

            if metric not in df.columns:
                continue

            s = pd.to_numeric(df[metric], errors="coerce").dropna()

            # Very negative values
            low = s[s < LOW_THRESHOLD]

            for value, count in low.value_counts().items():
                extreme_values[metric][float(value)] += int(count)

            # Very high values
            high = s[s > HIGH_THRESHOLD]

            for value, count in high.value_counts().items():
                extreme_values[metric][float(value)] += int(count)

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

# --------------------------------------------------
# Create output
# --------------------------------------------------

rows = []

for metric in METRICS:

    for value, count in extreme_values[metric].items():

        rows.append({
            "metric": metric,
            "value": value,
            "count": count
        })

result = pd.DataFrame(rows)

if not result.empty:
    result = result.sort_values(
        ["metric", "count"],
        ascending=[True, False]
    )

output_file = OUTPUT_DIR / "exact_extreme_values.csv"
result.to_csv(output_file, index=False)

print()
print("=" * 70)
print("EXACT EXTREME VALUES")
print("=" * 70)

if result.empty:
    print("No extreme values found.")
else:
    print(result.to_string(index=False))

print()
print("Saved:")
print(output_file)