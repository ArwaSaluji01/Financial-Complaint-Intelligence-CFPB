
import pandas as pd

from src.features.text_preprocessing import (
    clean_narrative,
    make_duplicate_key,
    is_valid_narrative
)

from src.data.prepare_dataset import build_task_dataset

from src.data.splitting import split_dataset


def create_sample_dataframe():
    return pd.DataFrame({
        "text": [
            "My credit card payment failed.",
            "My credit card payment failed.",
            "I have a debt collection problem.",
            "My mortgage payment is incorrect.",
            "A bank charged me unexpected fees.",
            "A student loan payment is missing."
        ],
        "product": [
            "Credit card",
            "Credit card",
            "Debt collection",
            "Mortgage",
            "Checking account",
            "Student loan"
        ],
        "issue": [
            "Payment issue",
            "Payment issue",
            "Debt issue",
            "Mortgage issue",
            "Fee issue",
            "Loan issue"
        ],
        "complaint_id": [
            "1", "2", "3", "4", "5", "6"
        ],
        "date_received": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06"
        ])
    })


def test_clean_narrative_whitespace():
    result = clean_narrative(
        "  My   payment \n failed.  "
    )

    assert result == "My payment failed."


def test_duplicate_key_case_insensitive():
    first = make_duplicate_key(
        "Credit Card PAYMENT"
    )

    second = make_duplicate_key(
        "credit card payment"
    )

    assert first == second


def test_empty_narrative_is_invalid():
    assert is_valid_narrative("") is False
    assert is_valid_narrative("   ") is False


def test_placeholder_narrative_is_invalid():
    assert is_valid_narrative("N/A") is False


def test_valid_narrative():
    assert is_valid_narrative(
        "My bank charged an unexpected fee."
    ) is True


def test_duplicate_narratives_are_grouped():
    df = create_sample_dataframe()

    from src.features.text_preprocessing import make_duplicate_key

    df["group_id"] = df["text"].map(
        make_duplicate_key
    )

    task_df, metadata = build_task_dataset(
        df,
        target_column="product",
        min_class_count=1
    )

    assert len(task_df) == 5
    assert metadata["conflicting_groups_excluded"] == 0


def test_split_has_no_group_overlap():
    rows = []

    for label in ["A", "B"]:

        for index in range(50):

            rows.append({
                "text": f"Complaint {label} {index}",
                "label": label,
                "group_id": f"{label}_{index}"
            })

    df = pd.DataFrame(rows)

    train, validation, test = split_dataset(df)

    train_groups = set(train["group_id"])
    validation_groups = set(validation["group_id"])
    test_groups = set(test["group_id"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_conflicting_product_labels_are_excluded():
    df = pd.DataFrame({
        "text": [
            "Same complaint.",
            "Same complaint.",
            "Different complaint."
        ],
        "product": [
            "Credit card",
            "Mortgage",
            "Student loan"
        ],
        "issue": [
            "Payment issue",
            "Payment issue",
            "Loan issue"
        ],
        "complaint_id": ["1", "2", "3"],
        "date_received": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03"
        ]),
        "group_id": [
            "duplicate_group",
            "duplicate_group",
            "unique_group"
        ]
    })

    task_df, metadata = build_task_dataset(
        df,
        target_column="product",
        min_class_count=1
    )

    assert len(task_df) == 1
    assert metadata["conflicting_groups_excluded"] == 1


def test_issue_classification_can_keep_product_conflicts():
    df = pd.DataFrame({
        "text": [
            "Same complaint.",
            "Same complaint."
        ],
        "product": [
            "Credit card",
            "Mortgage"
        ],
        "issue": [
            "Payment issue",
            "Payment issue"
        ],
        "complaint_id": ["1", "2"],
        "date_received": pd.to_datetime([
            "2026-01-01",
            "2026-01-02"
        ]),
        "group_id": [
            "duplicate_group",
            "duplicate_group"
        ]
    })

    task_df, metadata = build_task_dataset(
        df,
        target_column="issue",
        min_class_count=1
    )

    assert len(task_df) == 1
    assert metadata["conflicting_groups_excluded"] == 0
