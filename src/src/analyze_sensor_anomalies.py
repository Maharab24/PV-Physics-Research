from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/cleaned_final/system33_daily")
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

print("=" * 80)
print("SYSTEM 33 SENSOR ANOMALY ANALYSIS")
print("=" * 80)

files = sorted(INPUT_DIR.rglob("*.parquet"))

print(f"Input files: {len(files)}")

records = []

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    for metric in METRICS:

        if metric not in df.columns:
            continue

        s = pd.to_numeric(df[metric], errors="coerce").dropna()

        if len(s) == 0:
            continue

        records.append({
            "metric": metric,
            "file": str(file),
            "count": len(s),
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "negative": (s < 0).sum(),
            "zero": (s == 0).sum(),
            "gt_3000": (s > 3000).sum(),
            "gt_5000": (s > 5000).sum(),
            "gt_10000": (s > 10000).sum(),
        })

    if i % 500 == 0:
        print(f"Processed {i:,}/{len(files):,}")

result = pd.DataFrame(records)

result.to_csv(
    OUTPUT_DIR / "sensor_anomaly_by_file.csv",
    index=False
)

# ------------------------------------------------------------------
# Global summary
# ------------------------------------------------------------------

summary = (
    result
    .groupby("metric")
    .agg(
        observations=("count", "sum"),
        min_value=("min", "min"),
        max_value=("max", "max"),
        mean_of_file_means=("mean", "mean"),
        negative_count=("negative", "sum"),
        zero_count=("zero", "sum"),
        gt_3000=("gt_3000", "sum"),
        gt_5000=("gt_5000", "sum"),
        gt_10000=("gt_10000", "sum"),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "sensor_anomaly_summary.csv",
    index=False
)

print()
print("=" * 80)
print("GLOBAL SENSOR SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))

# ------------------------------------------------------------------
# Explicit suspicious-value search
# ------------------------------------------------------------------

SUSPICIOUS_RULES = {
    "ambient_temp": lambda s: (s > 80) | (s < -50),
    "dc_voltage": lambda s: (s > 1000) | (s < 0),
    "dc_current": lambda s: (s > 500) | (s < 0),
    "ac_voltage": lambda s: (s > 400) | (s < 0),
    "ac_current": lambda s: (s > 500) | (s < 0),
    "power_factor": lambda s: (s.abs() > 1),
    "ac_power": lambda s: (s.abs() > 10000) | (s < -1000),
    "dc_power": lambda s: (s.abs() > 10000),
    "battery_voltage": lambda s: (s > 500) | (s < 0),
}

suspicious_records = []

print()
print("=" * 80)
print("SEARCHING FOR PHYSICALLY SUSPICIOUS VALUES")
print("=" * 80)

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    for metric, rule in SUSPICIOUS_RULES.items():

        if metric not in df.columns:
            continue

        mask = rule(df[metric])

        if mask.any():

            bad = df.loc[mask, ["measured_on", metric]].copy()

            bad["metric"] = metric
            bad["source_file"] = str(file)

            suspicious_records.append(bad)

    if i % 500 == 0:
        print(f"Checked {i:,}/{len(files):,}")

if suspicious_records:

    suspicious = pd.concat(
        suspicious_records,
        ignore_index=True
    )

    suspicious = suspicious[
        ["source_file", "measured_on", "metric"]
        + [c for c in suspicious.columns if c not in
           ["source_file", "measured_on", "metric"]]
    ]

    suspicious.to_csv(
        OUTPUT_DIR / "sensor_suspicious_values.csv",
        index=False
    )

    print()
    print(f"Suspicious observations: {len(suspicious):,}")

    print()
    print(
        suspicious.groupby("metric")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

else:

    print("No suspicious observations found.")

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Saved:\n"
    f"  {OUTPUT_DIR / 'sensor_anomaly_summary.csv'}\n"
    f"  {OUTPUT_DIR / 'sensor_anomaly_by_file.csv'}\n"
    f"  {OUTPUT_DIR / 'sensor_suspicious_values.csv'}"
)