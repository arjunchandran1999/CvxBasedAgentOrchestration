from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TrainedRouterModel:
    """
    Picklable learned router model.

    - kind="tfidf": uses sklearn TfidfVectorizer + LogisticRegression
    - kind="sbert": uses sentence-transformers embeddings + LogisticRegression (encoder loaded at runtime)
    """

    strong_model: str
    weak_model: str
    kind: Literal["tfidf", "sbert"]
    vectorizer: Any | None = None
    clf: Any | None = None
    embedding_model: str | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TrainedRouterModel":
        with path.open("rb") as f:
            return pickle.load(f)

    def predict_proba_strong(self, prompt: str) -> float:
        """
        Return P(strong wins) in [0,1].
        """
        if self.kind == "tfidf":
            if self.vectorizer is None or self.clf is None:
                return 0.5
            Xp = self.vectorizer.transform([prompt])
            proba = self.clf.predict_proba(Xp)[0]
            return float(proba[1])

        # SBERT
        if self.clf is None or not self.embedding_model:
            return 0.5
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing router deps (sentence-transformers). Install with: pip install -e '.[router]'") from e
        enc = SentenceTransformer(self.embedding_model)
        xp = enc.encode([prompt], normalize_embeddings=True, show_progress_bar=False)
        proba = self.clf.predict_proba(xp)[0]
        return float(proba[1])


def train_tfidf_logreg(
    *,
    prompts: list[str],
    labels_strong_wins: list[int],
    strong_model: str,
    weak_model: str,
) -> TrainedRouterModel:
    """
    Train a simple router scorer using TF-IDF features + logistic regression.
    This is a lightweight, reproducible baseline that approximates RouteLLM-style
    learned routing (but trained on in-domain preference labels).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing router deps. Install with: pip install -e '.[router]'") from e

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vec.fit_transform(prompts)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X, labels_strong_wins)
    return TrainedRouterModel(strong_model=strong_model, weak_model=weak_model, kind="tfidf", vectorizer=vec, clf=clf)


def train_sentence_transformer_logreg(
    *,
    prompts: list[str],
    labels_strong_wins: list[int],
    strong_model: str,
    weak_model: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TrainedRouterModel:
    """
    Train a router using sentence-transformer embeddings + logistic regression.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing router deps. Install with: pip install -e '.[router]'") from e

    enc = SentenceTransformer(embedding_model)
    X = enc.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X, labels_strong_wins)
    return TrainedRouterModel(
        strong_model=strong_model,
        weak_model=weak_model,
        kind="sbert",
        vectorizer=None,
        clf=clf,
        embedding_model=embedding_model,
    )

