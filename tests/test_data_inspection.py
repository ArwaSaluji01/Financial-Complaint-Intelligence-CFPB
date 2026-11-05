
from scripts.inspect_cfpb_data import (
    normalize_column,
    find_column
)


def test_normalize_column():
    assert normalize_column(
        "Consumer Complaint Narrative"
    ) == "consumer_complaint_narrative"


def test_find_narrative_column():
    columns = [
        "Date received",
        "Product",
        "Consumer complaint narrative"
    ]

    result = find_column(
        columns,
        ["consumer_complaint_narrative"]
    )

    assert result == "Consumer complaint narrative"


def test_find_missing_column():
    columns = [
        "Date received",
        "Product"
    ]

    result = find_column(
        columns,
        ["consumer_complaint_narrative"]
    )

    assert result is None


def test_column_matching_is_case_insensitive():
    columns = ["PRODUCT", "ISSUE"]

    result = find_column(
        columns,
        ["product"]
    )

    assert result == "PRODUCT"
