from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw/system33")
OUTPUT_DIR = Path("data/processed/system33_daily")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRIC MAPPING
# ============================================================

METRICS = {
    582: "irradiance",
    583: "dc_power",
    584: "ac_power",
    585: "dc_voltage",
    586: "dc_current",
    587: "ac_voltage",
    588: "ac_current",
    589: "ambient_temp",
    590: "module_temp_1",
    591: "module_temp_2",
    592: "module_temp_3",
    593: "inverter_temp",
    594: "panel_temp",
    595: "battery_voltage",
    596: "power_factor",
}

METRIC_IDS = set(METRICS.keys())


# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    DATA_DIR.rglob("*.parquet")
)

print(
    f"Found {len(files)} Parquet files"
)


# ============================================================
# PROCESS FILE BY FILE
# ============================================================

total_input_rows = 0
total_filtered_rows = 0
total_output_rows = 0

failed_files = []


for i, file in enumerate(files, start=1):

    try:

        # ----------------------------------------------------
        # Read ONE daily file
        # ----------------------------------------------------

        df = pd.read_parquet(file)

        total_input_rows += len(df)

        # ----------------------------------------------------
        # Keep only research metrics
        # ----------------------------------------------------

        df = df[
            df["metric_id"].isin(METRIC_IDS)
        ].copy()

        if df.empty:
            continue

        total_filtered_rows += len(df)

        # ----------------------------------------------------
        # Map metric ID → variable name
        # ----------------------------------------------------

        df["variable"] = df[
            "metric_id"
        ].map(METRICS)

        # ----------------------------------------------------
        # Sentinel values
        # ----------------------------------------------------

        sentinel_mask = (
            df["value"] == -99999
        )

        sentinel_count = int(
            sentinel_mask.sum()
        )

        if sentinel_count > 0:

            df.loc[
                sentinel_mask,
                "value"
            ] = pd.NA

        # ----------------------------------------------------
        # Long → Wide
        # ----------------------------------------------------

        wide = (
            df.pivot_table(
                index="measured_on",
                columns="variable",
                values="value",
                aggfunc="mean"
            )
            .reset_index()
        )

        wide.columns.name = None

        wide = wide.sort_values(
            "measured_on"
        )

        total_output_rows += len(wide)

        # ----------------------------------------------------
        # Preserve original date structure
        # ----------------------------------------------------

        relative = file.relative_to(
            DATA_DIR
        )

        output_path = (
            OUTPUT_DIR
            / relative
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save daily wide dataset
        # ----------------------------------------------------

        wide.to_parquet(
            output_path,
            index=False,
            compression="snappy"
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            i % 100 == 0
            or i == len(files)
        ):

            print(
                f"[{i}/{len(files)}] "
                f"Processed | "
                f"Input rows: {total_input_rows:,} | "
                f"Filtered: {total_filtered_rows:,} | "
                f"Wide rows: {total_output_rows:,}"
            )

    except Exception as e:

        print(
            f"\nERROR: {file}"
        )

        print(e)

        failed_files.append(
            str(file)
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    f"Files found:          {len(files):,}"
)

print(
    f"Files failed:         {len(failed_files):,}"
)

print(
    f"Input rows:           {total_input_rows:,}"
)

print(
    f"Filtered rows:        {total_filtered_rows:,}"
)

print(
    f"Wide timestamp rows:  {total_output_rows:,}"
)

print(
    f"Output directory:     {OUTPUT_DIR}"
)


# ============================================================
# FAILED FILES
# ============================================================

if failed_files:

    failed_file = (
        OUTPUT_DIR.parent
        / "failed_files.txt"
    )

    with open(
        failed_file,
        "w",
        encoding="utf-8"
    ) as f:

        for path in failed_files:
            f.write(path + "\n")

    print(
        f"\nFailed files saved to:"
    )

    print(failed_file)