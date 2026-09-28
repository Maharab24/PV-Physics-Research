from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/processed/system33_daily")
OUTPUT_DIR = Path("data/cleaned_final/system33_daily")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

print("=" * 80)
print("FINAL SYSTEM 33 CLEANING")
print("=" * 80)
print(f"Input files : {len(files):,}")
print(f"Output dir  : {OUTPUT_DIR}")
print()

stats = {
    "rows": 0,

    "irradiance_negative_to_nan": 0,
    "irradiance_small_negative_to_zero": 0,
    "irradiance_7999": 0,

    "module_temp_1_7999": 0,
    "module_temp_2_7999": 0,
    "module_temp_3_7999": 0,

    "module_temp_1_3276_7": 0,
    "module_temp_2_3276_7": 0,
    "module_temp_3_3276_7": 0,

    "other_7999": 0,

    "missing_irradiance_files": 0,
}

for i, file in enumerate(files, 1):

    try:
        df = pd.read_parquet(file)

        stats["rows"] += len(df)

        # ---------------------------------------------------------
        # IRRADIANCE
        # ---------------------------------------------------------

        if "irradiance" in df.columns:

            irr = pd.to_numeric(
                df["irradiance"],
                errors="coerce"
            )

            # Explicit sentinel
            sentinel_7999 = irr == -7999

            stats["irradiance_7999"] += int(
                sentinel_7999.sum()
            )

            # Strong negative anomaly
            strong_negative = irr < -10

            stats["irradiance_negative_to_nan"] += int(
                strong_negative.sum()
            )

            # Small negative values
            small_negative = (
                (irr >= -10) &
                (irr < 0)
            )

            stats["irradiance_small_negative_to_zero"] += int(
                small_negative.sum()
            )

            # Apply rules
            irr = irr.mask(strong_negative, np.nan)
            irr = irr.mask(small_negative, 0.0)

            df["irradiance"] = irr

        else:
            stats["missing_irradiance_files"] += 1

        # ---------------------------------------------------------
        # MODULE TEMPERATURES
        # ---------------------------------------------------------

        for col in [
            "module_temp_1",
            "module_temp_2",
            "module_temp_3",
        ]:

            if col not in df.columns:
                continue

            s = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            # -7999 sentinel
            mask_7999 = s == -7999

            stats[f"{col}_7999"] += int(
                mask_7999.sum()
            )

            # 3276.7 invalid sensor value
            mask_3276 = s == 3276.7

            stats[f"{col}_3276_7"] += int(
                mask_3276.sum()
            )

            s = s.mask(mask_7999, np.nan)
            s = s.mask(mask_3276, np.nan)

            df[col] = s

        # ---------------------------------------------------------
        # OTHER METRICS: ONLY REMOVE EXPLICIT -7999
        # ---------------------------------------------------------

        protected = {
            "irradiance",
            "module_temp_1",
            "module_temp_2",
            "module_temp_3",
        }

        for col in df.columns:

            if col in protected:
                continue

            if col in [
                "measured_on",
                "utc_measured_on",
            ]:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            mask = df[col] == -7999

            stats["other_7999"] += int(mask.sum())

            if mask.any():
                df.loc[mask, col] = np.nan

        # ---------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------

        relative = file.relative_to(INPUT_DIR)
        output_file = OUTPUT_DIR / relative

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_parquet(
            output_file,
            index=False
        )

    except Exception as e:

        print()
        print(f"ERROR: {file}")
        print(e)

    if i % 500 == 0:
        print(
            f"Processed {i:,}/{len(files):,}"
        )

# -------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------

print()
print("=" * 80)
print("FINAL CLEANING SUMMARY")
print("=" * 80)

for key, value in stats.items():
    print(f"{key:40s}: {value:,}")

print()
print("Saved final cleaned dataset:")
print(OUTPUT_DIR)