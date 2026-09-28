import boto3
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict

SYSTEM_ID = "33"

s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED)
)

prefix = f"pvdaq/parquet/pvdata/system_id={SYSTEM_ID}/"

paginator = s3.get_paginator("list_objects_v2")

year_files = defaultdict(int)
year_size = defaultdict(int)

total_files = 0
total_size = 0

for page in paginator.paginate(
    Bucket="oedi-data-lake",
    Prefix=prefix
):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if not key.endswith(".parquet"):
            continue

        total_files += 1
        total_size += obj["Size"]

        parts = key.split("/")

        for part in parts:
            if part.startswith("year="):
                year = part.replace("year=", "")
                year_files[year] += 1
                year_size[year] += obj["Size"]
                break

print(f"System: {SYSTEM_ID}")
print(f"Total daily files: {total_files}")
print(f"Total size: {total_size / (1024**3):.2f} GB")

print("\nYear-wise coverage:")
for year in sorted(year_files):
    print(
        f"{year}: "
        f"{year_files[year]:5d} files | "
        f"{year_size[year] / (1024**3):.2f} GB"
    )