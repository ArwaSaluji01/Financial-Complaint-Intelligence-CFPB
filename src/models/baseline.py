from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_baseline_model(config):
    tfidf_config = config["tfidf"]
    classifier_config = config["classifier"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=tfidf_config["max_features"],
                    ngram_range=tuple(tfidf_config["ngram_range"]),
                    min_df=tfidf_config["min_df"],
                    sublinear_tf=tfidf_config["sublinear_tf"],
                    strip_accents=tfidf_config["strip_accents"],
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=classifier_config["C"],
                    class_weight=classifier_config["class_weight"],
                    max_iter=classifier_config["max_iter"],
                    random_state=classifier_config["random_state"],
                ),
            ),
        ]
    )

    return model