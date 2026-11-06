from src.models.baseline import build_baseline_model


def test_baseline_model_can_train_and_predict():
    config = {
        "tfidf": {
            "max_features": 1000,
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
        "classifier": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    }

    model = build_baseline_model(config)

    texts = [
        "My credit card payment was incorrectly charged.",
        "I have an issue with my credit card billing.",
        "Debt collectors keep contacting me about a loan.",
        "A collection agency is demanding payment for a debt.",
    ]

    labels = [
        "Credit card",
        "Credit card",
        "Debt collection",
        "Debt collection",
    ]

    model.fit(texts, labels)

    predictions = model.predict(texts)

    assert len(predictions) == len(texts)
    assert set(predictions).issubset(set(labels))