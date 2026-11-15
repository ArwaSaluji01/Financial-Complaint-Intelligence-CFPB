import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def create_label_mapping(labels):
    """
    Create deterministic mappings between class names and integer IDs.
    """
    unique_labels = sorted(set(labels))

    label2id = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    return label2id, id2label


def compute_training_class_weights(labels, label2id):
    """
    Calculate balanced class weights using training labels only.
    """
    encoded_labels = np.array(
        [label2id[label] for label in labels]
    )

    classes = np.arange(len(label2id))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=encoded_labels,
    )

    return torch.tensor(weights, dtype=torch.float32)


def build_distilbert_model(model_name, label2id, id2label):
    """
    Load a pretrained DistilBERT sequence classification model.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    return model