
import json
import re
import sys
import zipfile

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"

PLACEHOLDERS = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "not provided",
    "not available",
    "redacted",
    "xxxx"
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def analyze_dataset(archive_path, chunk_size):
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Invalid archive.")

    records = []

    with zipfile.ZipFile(archive_path) as archive:

        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV, found {len(csv_files)}."
            )

        with archive.open(csv_files[0]) as csv_file:

            chunks = pd.read_csv(
                csv_file,
                chunksize=chunk_size,
                dtype="string"
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk.columns = [
                    column.strip()
                    for column in chunk.columns
                ]

                narrative = (
                    chunk["Consumer complaint narrative"]
                    .fillna("")
                    .str.strip()
                )

                product = chunk["Product"].fillna("").str.strip()
                issue = chunk["Issue"].fillna("").str.strip()

                mask = (
                    narrative.ne("")
                    & ~narrative.str.lower().isin(PLACEHOLDERS)
                    & product.ne("")
                    & issue.ne("")
                )

                selected = chunk.loc[
                    mask,
                    [
                        "Consumer complaint narrative",
                        "Product",
                        "Issue",
                        "Complaint ID",
                        "Date received"
                    ]
                ].copy()

                records.append(selected)

                print(
                    f"Chunk {chunk_number}: "
                    f"{len(selected):,} usable records"
                )

    if not records:
        raise ValueError("No usable records found.")

    df = pd.concat(
        records,
        ignore_index=True
    )

    df.columns = [
        "narrative",
        "product",
        "issue",
        "complaint_id",
        "date_received"
    ]

    df["normalized_narrative"] = (
        df["narrative"].map(normalize_text)
    )

    df["date_received"] = pd.to_datetime(
        df["date_received"],
        errors="coerce"
    )

    # Record-level distributions.
    product_records = (
        df["product"]
        .value_counts()
        .to_dict()
    )

    issue_records = (
        df["issue"]
        .value_counts()
        .to_dict()
    )

    # Group exact normalized narratives.
    grouped = df.groupby(
        "normalized_narrative",
        sort=False
    )

    unique_narratives = len(grouped)

    duplicate_groups = grouped.filter(
        lambda group: len(group) > 1
    )

    # Identify conflicting labels.
    conflicting_product_groups = 0
    conflicting_issue_groups = 0

    product_conflicts = []
    issue_conflicts = []

    for narrative, group in grouped:

        if len(group) <= 1:
            continue

        products = group["product"].unique().tolist()
        issues = group["issue"].unique().tolist()

        if len(products) > 1:
            conflicting_product_groups += 1

            product_conflicts.append({
                "narrative": narrative[:250],
                "products": products,
                "records": len(group)
            })

        if len(issues) > 1:
            conflicting_issue_groups += 1

            issue_conflicts.append({
                "narrative": narrative[:250],
                "issues": issues,
                "records": len(group)
            })

    # Unique narrative counts per product and issue.
    unique_df = df.drop_duplicates(
        subset=["normalized_narrative"]
    )

    unique_product_counts = (
        unique_df["product"]
        .value_counts()
        .to_dict()
    )

    unique_issue_counts = (
        unique_df["issue"]
        .value_counts()
        .to_dict()
    )

    # Product and issue joint distribution.
    joint_counts = (
        df.groupby(["product", "issue"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Monthly distribution.
    df["month"] = (
        df["date_received"]
        .dt.to_period("M")
        .astype("string")
    )

    monthly_counts = (
        df["month"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report = {
        "total_usable_records": len(df),
        "unique_normalized_narratives": unique_narratives,
        "repeated_narrative_records": (
            len(df) - unique_narratives
        ),
        "duplicate_narrative_groups": (
            int(len(duplicate_groups["normalized_narrative"].unique()))
            if not duplicate_groups.empty else 0
        ),
        "conflicting_product_groups": (
            conflicting_product_groups
        ),
        "conflicting_issue_groups": (
            conflicting_issue_groups
        ),
        "product_record_distribution": product_records,
        "product_unique_narrative_distribution": (
            unique_product_counts
        ),
        "issue_record_distribution": issue_records,
        "issue_unique_narrative_distribution": (
            unique_issue_counts
        ),
        "monthly_distribution": monthly_counts,
        "unique_product_count": len(product_records),
        "unique_issue_count": len(issue_records),
        "product_issue_top_30": (
            joint_counts.head(30).to_dict("records")
        ),
        "product_conflict_examples": product_conflicts[:10],
        "issue_conflict_examples": issue_conflicts[:10],
        "date_minimum": str(
            df["date_received"].min()
        ),
        "date_maximum": str(
            df["date_received"].max()
        )
    }

    return report


def main():
    config = load_config()

    archive_path = (
        PROJECT_ROOT / config["paths"]["raw_archive"]
    )

    report = analyze_dataset(
        archive_path,
        config["inspection"]["chunk_size"]
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "metrics"
        / "narrative_subset_analysis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("NARRATIVE SUBSET ANALYSIS")
    print("=" * 60)

    for key in [
        "total_usable_records",
        "unique_normalized_narratives",
        "repeated_narrative_records",
        "duplicate_narrative_groups",
        "conflicting_product_groups",
        "conflicting_issue_groups",
        "unique_product_count",
        "unique_issue_count",
        "date_minimum",
        "date_maximum"
    ]:
        print(f"{key}: {report[key]}")

    print("\nPRODUCT DISTRIBUTION")

    for label, count in (
        report["product_record_distribution"].items()
    ):
        print(f"{label}: {count:,}")

    print("\nUNIQUE NARRATIVES PER PRODUCT")

    for label, count in (
        report["product_unique_narrative_distribution"].items()
    ):
        print(f"{label}: {count:,}")

    print("\nISSUE DISTRIBUTION")

    for label, count in (
        report["issue_record_distribution"].items()
    ):
        print(f"{label}: {count:,}")

    print("\nMONTHLY DISTRIBUTION")

    for month, count in report["monthly_distribution"].items():
        print(f"{month}: {count:,}")

    print("\nTOP PRODUCT-ISSUE PAIRS")

    for item in report["product_issue_top_30"]:
        print(
            f"{item['product']} | "
            f"{item['issue']} | "
            f"{item['count']}"
        )

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Analysis failed:", error)
        sys.exit(1)
