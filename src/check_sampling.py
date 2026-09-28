import boto3
import pandas as pd

from botocore import UNSIGNED
from botocore.config import Config
from io import BytesIO


SYSTEM_ID = "1200"

s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED)
)

prefix = f"pvdaq/parquet/pvdata/system_id={SYSTEM_ID}/"

print(f"Searching System {SYSTEM_ID}...")

paginator = s3.get_paginator("list_objects_v2")

candidate_files = []

for page in paginator.paginate(
    Bucket="oedi-data-lake",
    Prefix=prefix
):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if key.endswith(".parquet"):
            candidate_files.append(key)

print(f"Total parquet files found: {len(candidate_files)}")

if not candidate_files:
    raise SystemExit("No parquet file found.")


# Ignore obviously abnormal years
valid_files = []

for key in candidate_files:

    parts = key.split("/")

    year_part = next(
        (p for p in parts if p.startswith("year=")),
        None
    )

    if year_part is None:
        continue

    year = int(year_part.split("=")[1])

    if 2000 <= year <= 2023:
        valid_files.append(key)


print(f"Files with year 2000-2023: {len(valid_files)}")


# Choose a file from the middle of the valid list
file_key = valid_files[len(valid_files) // 2]

print("\nSelected representative file:")
print(file_key)


obj = s3.get_object(
    Bucket="oedi-data-lake",
    Key=file_key
)

df = pd.read_parquet(
    BytesIO(obj["Body"].read())
)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMetric counts:")
print(
    df["metric_id"]
    .value_counts()
    .sort_index()
)

print("\nSampling interval by metric:")

for metric_id in sorted(df["metric_id"].unique()):

    temp = df[df["metric_id"] == metric_id].copy()
    temp = temp.sort_values("measured_on")

    if len(temp) < 2:
        continue

    delta = temp["measured_on"].diff().dropna()

    print(
        f"Metric {metric_id}: "
        f"{len(temp)} observations | "
        f"median interval = {delta.median()}"
    )