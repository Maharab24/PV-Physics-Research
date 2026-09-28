from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("results/system33_quality")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND PROCESSED FILES
# ============================================================

files = sorted(
    DATA_DIR.rglob("*.parquet")
)

print("=" * 70)
print("SYSTEM 33 DATA QUALITY ANALYSIS")
print("=" * 70)

print(
    f"Processed Parquet files: {len(files):,}"
)

if not files:
    raise FileNotFoundError(
        "No processed Parquet files found."
    )


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# GLOBAL COUNTERS
# ============================================================

total_rows = 0

missing_counts = {
    metric: 0
    for metric in METRICS
}

valid_counts = {
    metric: 0
    for metric in METRICS
}

negative_counts = {
    metric: 0
    for metric in METRICS
}

zero_counts = {
    metric: 0
    for metric in METRICS
}

sentinel_counts = {
    metric: 0
    for metric in METRICS
}

duplicate_timestamps = 0

all_min = {
    metric: np.inf
    for metric in METRICS
}

all_max = {
    metric: -np.inf
    for metric in METRICS
}

all_values = {
    metric: []
    for metric in METRICS
}


# ============================================================
# PROCESS FILES
# ============================================================

for i, file in enumerate(files, start=1):

    try:

        df = pd.read_parquet(file)

        total_rows += len(df)

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        df["measured_on"] = pd.to_datetime(
            df["measured_on"],
            errors="coerce"
        )

        duplicate_timestamps += int(
            df["measured_on"].duplicated().sum()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        for metric in METRICS:

            if metric not in df.columns:
                continue

            values = df[metric]

            # Missing
            missing_counts[metric] += int(
                values.isna().sum()
            )

            # Valid
            valid = values.dropna()

            valid_counts[metric] += len(valid)

            if valid.empty:
                continue

            # Sentinel values
            sentinel_counts[metric] += int(
                valid.isin(
                    [-99999, -9999, -999]
                ).sum()
            )

            # Negative
            negative_counts[metric] += int(
                (valid < 0).sum()
            )

            # Zero
            zero_counts[metric] += int(
                (valid == 0).sum()
            )

            # Min / max
            current_min = valid.min()
            current_max = valid.max()

            all_min[metric] = min(
                all_min[metric],
                current_min
            )

            all_max[metric] = max(
                all_max[metric],
                current_max
            )

            # Sample values for statistics
            if len(valid) > 0:

                sample = valid.sample(
                    min(5000, len(valid)),
                    random_state=42
                )

                all_values[metric].extend(
                    sample.tolist()
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            i % 250 == 0
            or i == len(files)
        ):

            print(
                f"Processed "
                f"{i:,}/{len(files):,}"
            )

    except Exception as e:

        print(
            f"ERROR: {file}"
        )

        print(e)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BASIC SUMMARY")
print("=" * 70)

print(
    f"Total timestamp rows: {total_rows:,}"
)

print(
    f"Duplicate timestamps: {duplicate_timestamps:,}"
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

missing_rows = []

for metric in METRICS:

    missing = missing_counts[metric]

    valid = valid_counts[metric]

    total = missing + valid

    percent = (
        missing / total * 100
        if total > 0
        else 0
    )

    missing_rows.append({
        "metric": metric,
        "missing_count": missing,
        "valid_count": valid,
        "total_count": total,
        "missing_percent": percent
    })


missing_df = pd.DataFrame(
    missing_rows
)

print("\n")
print("=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

print(
    missing_df.to_string(
        index=False
    )
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)


# ============================================================
# SENTINEL / NEGATIVE / ZERO ANALYSIS
# ============================================================

quality_rows = []

for metric in METRICS:

    quality_rows.append({
        "metric": metric,
        "sentinel_count":
            sentinel_counts[metric],
        "negative_count":
            negative_counts[metric],
        "zero_count":
            zero_counts[metric],
        "min":
            all_min[metric],
        "max":
            all_max[metric]
    })


quality_df = pd.DataFrame(
    quality_rows
)

print("\n")
print("=" * 70)
print("VALUE QUALITY ANALYSIS")
print("=" * 70)

print(
    quality_df.to_string(
        index=False
    )
)

quality_df.to_csv(
    OUTPUT_DIR / "value_quality.csv",
    index=False
)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

stats_rows = []

for metric in METRICS:

    values = pd.Series(
        all_values[metric]
    )

    if values.empty:
        continue

    stats_rows.append({
        "metric": metric,
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "25_percentile":
            values.quantile(0.25),
        "median":
            values.median(),
        "75_percentile":
            values.quantile(0.75),
        "max": values.max()
    })


stats_df = pd.DataFrame(
    stats_rows
)

print("\n")
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    stats_df.to_string(
        index=False
    )
)

stats_df.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv",
    index=False
)


# ============================================================
# PHYSICAL RANGE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("PHYSICAL RANGE CHECK")
print("=" * 70)


# These are screening thresholds.
# They are NOT final cleaning rules.

physical_checks = []

for metric in METRICS:

    min_value = all_min[metric]
    max_value = all_max[metric]

    physical_checks.append({
        "metric": metric,
        "observed_min": min_value,
        "observed_max": max_value
    })


physical_df = pd.DataFrame(
    physical_checks
)

print(
    physical_df.to_string(
        index=False
    )
)

physical_df.to_csv(
    OUTPUT_DIR / "physical_range_check.csv",
    index=False
)


# ============================================================
# IRRADIANCE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("IRRADIANCE ANALYSIS")
print("=" * 70)

irr_negative = negative_counts[
    "irradiance"
]

irr_zero = zero_counts[
    "irradiance"
]

irr_above_1200 = 0
irr_above_1500 = 0

# We need another pass for these
for file in files:

    df = pd.read_parquet(file)

    if "irradiance" not in df.columns:
        continue

    values = df[
        "irradiance"
    ].dropna()

    irr_above_1200 += int(
        (values > 1200).sum()
    )

    irr_above_1500 += int(
        (values > 1500).sum()
    )


print(
    "Negative irradiance:",
    irr_negative
)

print(
    "Zero irradiance:",
    irr_zero
)

print(
    "Irradiance > 1200 W/m²:",
    irr_above_1200
)

print(
    "Irradiance > 1500 W/m²:",
    irr_above_1500
)


# ============================================================
# DC POWER ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("DC POWER ANALYSIS")
print("=" * 70)

print(
    "Negative DC power:",
    negative_counts["dc_power"]
)

print(
    "Zero DC power:",
    zero_counts["dc_power"]
)


# ============================================================
# GRAPH 1
# IRRADIANCE DISTRIBUTION
# ============================================================

irr_samples = pd.Series(
    all_values["irradiance"]
).dropna()

if not irr_samples.empty:

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        irr_samples,
        bins=60
    )

    plt.xlabel(
        "POA Irradiance (W/m²)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "System 33 — POA Irradiance Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "irradiance_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 2
# DC POWER DISTRIBUTION
# ============================================================

power_samples = pd.Series(
    all_values["dc_power"]
).dropna()

if not power_samples.empty:

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        power_samples,
        bins=60
    )

    plt.xlabel(
        "DC Power (W)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "System 33 — DC Power Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "dc_power_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_DIR}"
)