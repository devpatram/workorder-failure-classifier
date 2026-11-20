"""Baseline 2: TF-IDF features and logistic regression, one model for the component and one for the failure.

Word and character n-grams, so "brg" and "bearing" and small typos still share features. Trained on train.csv only.
"""
from __future__ import annotations


class TfidfClassifier:
    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence
        self.models = {}

    @staticmethod
    def _pipeline():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, make_pipeline

        features = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)),
        ])
        return make_pipeline(features, LogisticRegression(max_iter=2000, C=5.0))

    def fit(self, texts: list[str], components: list[str], failures: list[str]) -> "TfidfClassifier":
        for target, labels in (("component", components), ("failure", failures)):
            self.models[target] = self._pipeline().fit(texts, labels)
        return self

    def classify(self, text: str) -> dict:
        out = {}
        for target, model in self.models.items():
            proba = model.predict_proba([text])[0]
            best = proba.argmax()
            out[target] = model.classes_[best] if proba[best] >= self.min_confidence else None
            out[f"{target}_confidence"] = round(float(proba[best]), 3)
        return out
