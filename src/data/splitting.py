
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


RANDOM_STATE = 42


def split_dataset(
    df,
    label_column="label",
    group_column="group_id"
):
    """
    Split data into approximately:

    Train:      64%
    Validation: 16%
    Test:       20%

    Duplicate groups are never separated.
    """

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    if df[label_column].isna().any():
        raise ValueError("Missing labels detected.")

    if df[group_column].isna().any():
        raise ValueError("Missing group identifiers detected.")

    if df[group_column].duplicated().any():
        raise ValueError(
            "Expected one representative record per group."
        )

    class_counts = df[label_column].value_counts()

    if (class_counts < 5).any():
        raise ValueError(
            "Every retained class needs at least five groups "
            "for the five-fold stratified split."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    X = df["text"]
    y = df[label_column]
    groups = df[group_column]

    # First split: 80% development, 20% test.
    development_indices, test_indices = next(
        splitter.split(X, y, groups)
    )

    development_df = (
        df.iloc[development_indices]
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[test_indices]
        .reset_index(drop=True)
    )

    # Second split: 80% train, 20% validation
    # of the development portion.
    second_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE + 1
    )

    train_indices, validation_indices = next(
        second_splitter.split(
            development_df["text"],
            development_df[label_column],
            development_df[group_column]
        )
    )

    train_df = (
        development_df.iloc[train_indices]
        .reset_index(drop=True)
    )

    validation_df = (
        development_df.iloc[validation_indices]
        .reset_index(drop=True)
    )

    # Verify that no group appears in multiple splits.
    train_groups = set(train_df[group_column])
    validation_groups = set(validation_df[group_column])
    test_groups = set(test_df[group_column])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    return train_df, validation_df, test_df


def get_split_statistics(
    train_df,
    validation_df,
    test_df,
    label_column="label"
):
    """
    Generate split statistics for reporting.
    """

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df
    }

    statistics = {}

    for name, split_df in splits.items():

        statistics[name] = {
            "record_count": len(split_df),
            "class_distribution": (
                split_df[label_column]
                .value_counts()
                .to_dict()
            )
        }

    return statistics
