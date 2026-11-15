import torch

from src.models.distilbert_model import (
    compute_training_class_weights,
    create_label_mapping,
)


def test_label_mapping_is_deterministic():
    labels = [
        "Credit card",
        "Debt collection",
        "Credit card",
    ]

    label2id, id2label = create_label_mapping(labels)

    assert label2id == {
        "Credit card": 0,
        "Debt collection": 1,
    }

    assert id2label == {
        0: "Credit card",
        1: "Debt collection",
    }


def test_class_weights_are_valid():
    labels = [
        "Credit card",
        "Credit card",
        "Credit card",
        "Debt collection",
    ]

    label2id, _ = create_label_mapping(labels)

    weights = compute_training_class_weights(
        labels,
        label2id,
    )

    assert isinstance(weights, torch.Tensor)
    assert len(weights) == 2
    assert torch.all(weights > 0)

    # The less frequent class should receive a higher weight.
    assert weights[label2id["Debt collection"]] > (
        weights[label2id["Credit card"]]
    )