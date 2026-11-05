
import json
import re
import sys
import zipfile

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


COLUMN_ALIASES = {
    "narrative": [
        "consumer_complaint_narrative",
        "complaint_narrative",
        "narrative"
    ],
    "product": [
        "product"
    ],
    "issue": [
        "issue"
    ],
    "complaint_id": [
        "complaint_id"
    ],
    "date_received": [
        "date_received"
    ]
}

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


def normalize_column(name):
    name = str(name).strip().lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    return name.strip("_")


def find_column(columns, aliases):
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    return None


def inspect_archive(archive_path, config):
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}"
        )

    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Invalid ZIP archive.")

    with zipfile.ZipFile(archive_path, "r") as archive:

        csv_files = [
            item for item in archive.namelist()
            if item.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "No CSV file found inside the archive."
            )

        if len(csv_files) > 1:
            print("Multiple CSV files found:")
            for name in csv_files:
                print("-", name)

            raise ValueError(
                "Expected one CSV file. "
                "Inspect archive contents before proceeding."
            )

        csv_name = csv_files[0]

        print("CSV file:", csv_name)

        # Discover actual fields before assuming their names.
        with archive.open(csv_name) as csv_file:
            header = pd.read_csv(
                csv_file,
                nrows=0
            )

        original_columns = header.columns.tolist()

        print("\nDiscovered columns:")

        for column in original_columns:
            print("-", repr(column))

        columns = {
            key: find_column(
                original_columns,
                aliases
            )
            for key, aliases in COLUMN_ALIASES.items()
        }

        print("\nDetected important fields:")

        for key, value in columns.items():
            print(f"{key}: {value}")

        if columns["narrative"] is None:
            raise ValueError(
                "No complaint narrative field found. "
                "Cannot proceed with NLP classification."
            )

        if columns["product"] is None:
            raise ValueError(
                "No product label field found."
            )

        chunk_size = config["inspection"]["chunk_size"]

        missing_counts = Counter()
        product_counts = Counter()
        issue_counts = Counter()

        total_records = 0
        narrative_missing = 0
        narrative_placeholders = 0
        narrative_exact_duplicates = 0

        duplicate_ids = 0
        seen_ids = set()
        seen_narratives = set()

        narrative_lengths = []
        date_min = None
        date_max = None

        text_column = columns["narrative"]
        product_column = columns["product"]
        issue_column = columns["issue"]
        id_column = columns["complaint_id"]
        date_column = columns["date_received"]

        with archive.open(csv_name) as csv_file:

            chunks = pd.read_csv(
                csv_file,
                chunksize=chunk_size,
                dtype="string",
                on_bad_lines="error"
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                total_records += len(chunk)

                # Missing-value inspection for every field.
                for column in original_columns:

                    values = chunk[column].astype("string")

                    missing = (
                        values.isna()
                        | values.str.strip().eq("").fillna(True)
                    )

                    missing_counts[column] += int(
                        missing.sum()
                    )

                # Product distribution.
                if product_column:
                    values = (
                        chunk[product_column]
                        .fillna("<MISSING>")
                        .str.strip()
                    )

                    product_counts.update(values.tolist())

                # Issue distribution.
                if issue_column:
                    values = (
                        chunk[issue_column]
                        .fillna("<MISSING>")
                        .str.strip()
                    )

                    issue_counts.update(values.tolist())

                # Complaint narrative inspection.
                narratives = chunk[text_column].fillna("").str.strip()

                for narrative in narratives:

                    if not narrative:
                        narrative_missing += 1
                        continue

                    normalized = narrative.lower()

                    if normalized in PLACEHOLDERS:
                        narrative_placeholders += 1
                        continue

                    narrative_lengths.append(
                        len(narrative)
                    )

                    if normalized in seen_narratives:
                        narrative_exact_duplicates += 1
                    else:
                        seen_narratives.add(normalized)

                # Complaint ID duplicate inspection.
                if id_column:

                    ids = chunk[id_column].dropna().tolist()

                    for complaint_id in ids:

                        if complaint_id in seen_ids:
                            duplicate_ids += 1
                        else:
                            seen_ids.add(complaint_id)

                # Actual date coverage.
                if date_column:

                    dates = pd.to_datetime(
                        chunk[date_column],
                        errors="coerce"
                    ).dropna()

                    if not dates.empty:

                        chunk_min = dates.min()
                        chunk_max = dates.max()

                        date_min = (
                            chunk_min if date_min is None
                            else min(date_min, chunk_min)
                        )

                        date_max = (
                            chunk_max if date_max is None
                            else max(date_max, chunk_max)
                        )

                print(
                    f"Processed chunk {chunk_number} | "
                    f"Records processed: {total_records:,}"
                )

    # Build inspection report.
    missing_report = {}

    for column in original_columns:

        count = missing_counts[column]

        missing_report[column] = {
            "missing_count": count,
            "missing_percentage": round(
                count / total_records * 100, 2
            ) if total_records else 0
        }

    if narrative_lengths:

        length_statistics = {
            "count": len(narrative_lengths),
            "minimum": int(np.min(narrative_lengths)),
            "median": float(np.median(narrative_lengths)),
            "mean": round(
                float(np.mean(narrative_lengths)), 2
            ),
            "maximum": int(np.max(narrative_lengths)),
            "percentile_90": float(
                np.percentile(narrative_lengths, 90)
            )
        }

    else:
        length_statistics = {}

    top_categories = config["inspection"]["top_categories"]

    report = {
        "dataset": config["source"]["name"],
        "archive_label": config["source"]["archive_label"],
        "archive_file": archive_path.name,
        "total_records": total_records,
        "columns": original_columns,
        "detected_fields": columns,
        "missing_values": missing_report,
        "narrative_statistics": {
            "missing_narratives": narrative_missing,
            "placeholder_narratives": narrative_placeholders,
            "usable_narrative_count": len(narrative_lengths),
            "exact_duplicate_narratives": (
                narrative_exact_duplicates
            ),
            "length_statistics": length_statistics
        },
        "duplicate_complaint_ids": duplicate_ids,
        "product_distribution": dict(
            product_counts.most_common(top_categories)
        ),
        "issue_distribution": dict(
            issue_counts.most_common(top_categories)
        ),
        "date_coverage": {
            "minimum": str(date_min) if date_min is not None else None,
            "maximum": str(date_max) if date_max is not None else None
        }
    }

    return report


def save_report(report, json_path, text_path):
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    lines = []

    lines.append("=" * 60)
    lines.append("CFPB DATASET INSPECTION REPORT")
    lines.append("=" * 60)

    lines.append(
        f"\nDataset: {report['dataset']}"
    )

    lines.append(
        f"Archive: {report['archive_label']}"
    )

    lines.append(
        f"\nTotal records: {report['total_records']:,}"
    )

    lines.append("\nDISCOVERED COLUMNS")

    for column in report["columns"]:
        lines.append(f"- {column}")

    lines.append("\nDETECTED FIELDS")

    for key, value in report["detected_fields"].items():
        lines.append(f"{key}: {value}")

    lines.append("\nMISSING VALUES")

    for column, stats in report["missing_values"].items():
        lines.append(
            f"{column}: {stats['missing_count']:,} "
            f"({stats['missing_percentage']}%)"
        )

    lines.append("\nNARRATIVE STATISTICS")

    for key, value in report["narrative_statistics"].items():
        lines.append(f"{key}: {value}")

    lines.append("\nDUPLICATE COMPLAINT IDS")

    lines.append(
        str(report["duplicate_complaint_ids"])
    )

    lines.append("\nPRODUCT DISTRIBUTION")

    for label, count in report["product_distribution"].items():
        lines.append(f"{label}: {count:,}")

    lines.append("\nISSUE DISTRIBUTION")

    for label, count in report["issue_distribution"].items():
        lines.append(f"{label}: {count:,}")

    lines.append("\nDATE COVERAGE")

    for key, value in report["date_coverage"].items():
        lines.append(f"{key}: {value}")

    lines.append("\n" + "=" * 60)

    text_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "\n".join(lines))


def main():
    config = load_config()

    archive_path = (
        PROJECT_ROOT / config["paths"]["raw_archive"]
    )

    json_path = (
        PROJECT_ROOT / config["paths"]["inspection_report"]
    )

    text_path = (
        PROJECT_ROOT / config["paths"]["inspection_text"]
    )

    try:
        report = inspect_archive(
            archive_path,
            config
        )

        save_report(
            report,
            json_path,
            text_path
        )

        print("\nInspection completed successfully.")

        print("JSON report:", json_path)
        print("Text report:", text_path)

    except Exception as error:
        print("\nINSPECTION FAILED")
        print("Reason:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
