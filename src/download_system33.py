import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from tqdm import tqdm

BUCKET = "oedi-data-lake"
PREFIX = "pvdaq/parquet/pvdata/system_id=33/"

OUTPUT_DIR = Path("data/raw/system33")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED)
)

print("Finding System 33 files...")

files = []

paginator = s3.get_paginator("list_objects_v2")

for page in paginator.paginate(
    Bucket=BUCKET,
    Prefix=PREFIX
):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if key.endswith(".parquet"):
            files.append({
                "key": key,
                "size": obj["Size"]
            })

print(f"Total files: {len(files)}")

total_size = sum(x["size"] for x in files)

print(f"Total size: {total_size / (1024**3):.2f} GB")

print("\nDownloading...")

for item in tqdm(files):

    key = item["key"]

    relative_path = key.replace(PREFIX, "")
    local_path = OUTPUT_DIR / relative_path

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Skip already downloaded files
    if local_path.exists():
        continue

    s3.download_file(
        BUCKET,
        key,
        str(local_path)
    )

print("\nDownload completed.")
print(f"Location: {OUTPUT_DIR}")