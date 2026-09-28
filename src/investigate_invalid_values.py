from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/processed/system33_daily")

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

# Values we specifically want to investigate
SUSPECT_VALUES = [
    -99999,
    -9999,
    -999,
    -99,
    999,
    9999,
    99999,
    2011,
    2154,
    3276.7,
    129934,
    7809,
]

files = sorted(
    DATA_DIR.rglob("*.parquet")
)

print("=" * 70)
print("INVALID / EXTREME VALUE INVESTIGATION")
print("=" * 70)

print(
    f"Files: {len(files):,}"
)

results = []

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    for metric in METRICS:

        if metric not in df.columns:
            continue

        values = df[metric].dropna()

        if values.empty:
            continue

        # Suspect exact values
        for suspect in SUSPECT_VALUES:

            count = int(
                (values == suspect).sum()
            )

            if count > 0:

                results.append({
                    "metric": metric,
                    "type": "exact_value",
                    "value": suspect,
                    "count": count,
                    "file": str(file)
                })

        # Very extreme negative values
        count = int(
            (values < -1000).sum()
        )

        if count > 0:

            results.append({
                "metric": metric,
                "type": "less_than_-1000",
                "value": np.nan,
                "count": count,
                "file": str(file)
            })

        # Very extreme positive values
        count = int(
            (values > 3000).sum()
        )

        if count > 0:

            results.append({
                "metric": metric,
                "type": "greater_than_3000",
                "value": np.nan,
                "count": count,
                "file": str(file)
            })

    if (
        i % 500 == 0
        or i == len(files)
    ):

        print(
            f"Processed "
            f"{i:,}/{len(files):,}"
        )


result_df = pd.DataFrame(results)

output = Path(
    "results/system33_quality/"
    "invalid_value_details.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    output,
    index=False
)


print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if result_df.empty:

    print(
        "No suspicious values found."
    )

else:

    summary = (
        result_df
        .groupby(
            [
                "metric",
                "type",
                "value"
            ],
            dropna=False
        )["count"]
        .sum()
        .reset_index()
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

print("\nSaved:")
print(output)