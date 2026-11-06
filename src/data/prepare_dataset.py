
import hashlib
import zipfile

import pandas as pd

from src.features.text_preprocessing import (
    clean_narrative,
    make_duplicate_key,
    is_valid_narrative
)


REQUIRED_COLUMNS = [
    "Consumer complaint narrative",
    "Product",
    "Issue",
    "Complaint ID",
    "Date received"
]


def load_complaints_from_archive(
    archive_path,
    chunk_size=50000
):
    """
    Read the official CFPB archive in chunks.

    Only usable complaint narratives are retained.
    """

    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Invalid CFPB ZIP archive.")

    frames = []

    with zipfile.ZipFile(archive_path, "r") as archive:

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

            for chunk in chunks:

                chunk.columns = [
                    column.strip()
                    for column in chunk.columns
                ]

                missing_columns = [
                    column
                    for column in REQUIRED_COLUMNS
                    if column not in chunk.columns
                ]

                if missing_columns:
                    raise ValueError(
                        f"Missing required columns: {missing_columns}"
                    )

                chunk = chunk[REQUIRED_COLUMNS].copy()

                chunk["Consumer complaint narrative"] = (
                    chunk["Consumer complaint narrative"]
                    .fillna("")
                )

                chunk["Product"] = (
                    chunk["Product"]
                    .fillna("")
                    .str.strip()
                )

                chunk["Issue"] = (
                    chunk["Issue"]
                    .fillna("")
                    .str.strip()
                )

                mask = (
                    chunk["Consumer complaint narrative"]
                    .map(is_valid_narrative)
                    & chunk["Product"].ne("")
                    & chunk["Issue"].ne("")
                )

                chunk = chunk.loc[mask].copy()

                if not chunk.empty:
                    frames.append(chunk)

    if not frames:
        raise ValueError(
            "No usable complaint records found."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = df.rename(
        columns={
            "Consumer complaint narrative": "text",
            "Product": "product",
            "Issue": "issue",
            "Complaint ID": "complaint_id",
            "Date received": "date_received"
        }
    )

    df["text"] = df["text"].map(clean_narrative)

    df["date_received"] = pd.to_datetime(
        df["date_received"],
        errors="coerce"
    )

    # Remove records with invalid dates.
    df = df.dropna(
        subset=["date_received"]
    ).copy()

    # Exact normalized narrative key.
    df["duplicate_key"] = (
        df["text"].map(make_duplicate_key)
    )

    # Stable group identifier.
    df["group_id"] = df["duplicate_key"].map(
        lambda value: hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
    )

    return df.reset_index(drop=True)


def build_task_dataset(
    df,
    target_column,
    min_class_count=100
):
    """
    Create a deduplicated dataset for one classification task.

    Conflicting target labels for identical narratives
    are excluded from that task only.

    Minimum class support is measured using distinct
    normalized narrative groups.
    """

    if target_column not in {"product", "issue"}:
        raise ValueError(
            "target_column must be 'product' or 'issue'."
        )

    if min_class_count < 1:
        raise ValueError(
            "min_class_count must be positive."
        )

    records = []

    conflicting_groups = 0

    for group_id, group in df.groupby(
        "group_id",
        sort=False
    ):

        labels = group[target_column].dropna().unique()

        # Exclude groups with conflicting target labels.
        if len(labels) != 1:
            conflicting_groups += 1
            continue

        # One representative record per exact narrative.
        representative = (
            group.sort_values("date_received")
            .iloc[0]
        )

        records.append({
            "text": representative["text"],
            "label": labels[0],
            "complaint_id": representative["complaint_id"],
            "date_received": representative["date_received"],
            "duplicate_count": len(group),
            "group_id": group_id
        })

    task_df = pd.DataFrame(records)

    if task_df.empty:
        raise ValueError(
            f"No valid records for {target_column}."
        )

    # Support is measured after deduplication and
    # conflict removal.
    class_counts = task_df["label"].value_counts()

    retained_labels = class_counts[
        class_counts >= min_class_count
    ].index

    task_df = task_df[
        task_df["label"].isin(retained_labels)
    ].copy()

    task_df = task_df.reset_index(drop=True)

    if task_df.empty:
        raise ValueError(
            f"No classes meet the minimum support "
            f"for {target_column}."
        )

    metadata = {
        "task": target_column,
        "total_records_before_filtering": len(df),
        "unique_groups_before_filtering": df["group_id"].nunique(),
        "conflicting_groups_excluded": conflicting_groups,
        "minimum_class_support": min_class_count,
        "retained_class_count": task_df["label"].nunique(),
        "retained_record_count": len(task_df),
        "excluded_class_count": (
            len(class_counts) - len(retained_labels)
        ),
        "class_distribution": (
            task_df["label"].value_counts().to_dict()
        )
    }

    return task_df, metadata
