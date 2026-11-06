
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.data.prepare_dataset import (
    load_complaints_from_archive,
    build_task_dataset
)

from src.data.splitting import (
    split_dataset,
    get_split_statistics
)

CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_csv(df, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )


def save_json(data, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def prepare_task(
    df,
    task_name,
    output_directory,
    min_class_count
):
    print("\n" + "=" * 60)
    print(f"PREPARING {task_name.upper()} CLASSIFICATION")
    print("=" * 60)

    task_df, task_metadata = build_task_dataset(
        df,
        target_column=task_name,
        min_class_count=min_class_count
    )

    print("\nRetained records:", len(task_df))
    print("Retained classes:", task_df["label"].nunique())

    print("\nClass distribution:")

    for label, count in (
        task_df["label"].value_counts().items()
    ):
        print(f"{label}: {count:,}")

    train_df, validation_df, test_df = split_dataset(
        task_df
    )

    # Save complete deduplicated dataset.
    save_csv(
        task_df,
        output_directory / "all.csv"
    )

    # Save individual splits.
    save_csv(
        train_df,
        output_directory / "train.csv"
    )

    save_csv(
        validation_df,
        output_directory / "validation.csv"
    )

    save_csv(
        test_df,
        output_directory / "test.csv"
    )

    split_statistics = get_split_statistics(
        train_df,
        validation_df,
        test_df
    )

    metadata = {
        **task_metadata,
        "split_seed": 42,
        "splitting_strategy": (
            "Two-stage StratifiedGroupKFold"
        ),
        "splits": split_statistics
    }

    save_json(
        metadata,
        output_directory / "metadata.json"
    )

    print("\nSplit sizes:")

    for split_name, split_stats in (
        split_statistics.items()
    ):
        print(
            f"{split_name}: "
            f"{split_stats['record_count']:,}"
        )

    print("\nSaved files:", output_directory)

    return metadata


def main():
    config = load_config()

    archive_path = (
        PROJECT_ROOT / config["paths"]["raw_archive"]
    )

    output_root = (
        PROJECT_ROOT / "data" / "processed"
    )

    min_class_count = 100

    print("Loading CFPB complaint archive...")

    df = load_complaints_from_archive(
        archive_path,
        chunk_size=config["inspection"]["chunk_size"]
    )

    print("\nUsable records:", len(df))
    print("Unique narrative groups:", df["group_id"].nunique())

    print("\nDate coverage:")
    print("Minimum:", df["date_received"].min())
    print("Maximum:", df["date_received"].max())

    # Primary task.
    product_metadata = prepare_task(
        df=df,
        task_name="product",
        output_directory=output_root / "product",
        min_class_count=min_class_count
    )

    # Secondary task.
    issue_metadata = prepare_task(
        df=df,
        task_name="issue",
        output_directory=output_root / "issue",
        min_class_count=min_class_count
    )

    combined_metadata = {
        "dataset": config["source"]["name"],
        "archive_label": config["source"]["archive_label"],
        "minimum_class_support": min_class_count,
        "random_seed": 42,
        "tasks": {
            "product": product_metadata,
            "issue": issue_metadata
        }
    }

    save_json(
        combined_metadata,
        output_root / "preparation_metadata.json"
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nProduct classes:", product_metadata["retained_class_count"])
    print("Issue classes:", issue_metadata["retained_class_count"])

    print("\nProcessed data saved to:", output_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nDATA PREPARATION FAILED")
        print("Reason:", error)
        sys.exit(1)
