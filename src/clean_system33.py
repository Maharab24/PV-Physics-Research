from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("data/cleaned/system33_daily")

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

TEMPERATURE_COLUMNS = [
    "module_temp_1",
    "module_temp_2",
    "module_temp_3",
]

files = sorted(INPUT_DIR.rglob("*.parquet"))

print("=" * 70)
print("SYSTEM 33 DATA CLEANING")
print("=" * 70)
print(f"Input files : {len(files):,}")
print(f"Output dir  : {OUTPUT_DIR}")
print()

total_rows = 0
total_invalid = {}

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(file)

        total_rows += len(df)

        # --------------------------------------------------
        # 1. Convert known sentinel -7999 to NaN
        # --------------------------------------------------

        for col in METRICS:

            if col not in df.columns:
                continue

            mask = df[col] == -7999

            count = int(mask.sum())

            if count > 0:
                total_invalid[f"{col}: -7999"] = (
                    total_invalid.get(f"{col}: -7999", 0) + count
                )

                df.loc[mask, col] = np.nan

        # --------------------------------------------------
        # 2. Module-temperature overflow/sentinel
        # --------------------------------------------------

        for col in TEMPERATURE_COLUMNS:

            if col not in df.columns:
                continue

            mask = df[col] == 3276.7

            count = int(mask.sum())

            if count > 0:
                total_invalid[f"{col}: 3276.7"] = (
                    total_invalid.get(f"{col}: 3276.7", 0) + count
                )

                df.loc[mask, col] = np.nan

        # --------------------------------------------------
        # 3. Physically impossible negative irradiance
        # --------------------------------------------------

        if "irradiance" in df.columns:

            mask = df["irradiance"] < 0

            count = int(mask.sum())

            if count > 0:
                total_invalid["irradiance: negative"] = (
                    total_invalid.get(
                        "irradiance: negative", 0
                    ) + count
                )

                df.loc[mask, "irradiance"] = np.nan

        # --------------------------------------------------
        # Save using same directory structure
        # --------------------------------------------------

        relative_path = file.relative_to(INPUT_DIR)
        output_file = OUTPUT_DIR / relative_path

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_parquet(
            output_file,
            index=False
        )

    except Exception as e:

        print(f"\nERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(
            f"Processed {i:,}/{len(files):,}"
        )


# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print()
print("=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(f"Total rows processed: {total_rows:,}")
print()

for key, count in sorted(
    total_invalid.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{key:35s} {count:>10,}")

print()
print("Saved cleaned dataset to:")
print(OUTPUT_DIR)