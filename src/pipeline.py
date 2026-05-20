from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class PipelineArtifacts:
    classifier: Pipeline


def train_category_model(df: pd.DataFrame) -> PipelineArtifacts:
    X_train, _, y_train, _ = train_test_split(
        df["complaint_text"],
        df["category_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["category_label"],
    )

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("model", LogisticRegression(max_iter=800)),
        ]
    )
    clf.fit(X_train, y_train)
    return PipelineArtifacts(classifier=clf)


def sentiment_and_urgency(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    def score_sentiment(text: str) -> float:
        return analyzer.polarity_scores(text)["compound"]

    df = df.copy()
    df["sentiment_score"] = df["complaint_text"].apply(score_sentiment)
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.0, -0.20, 0.20, 1.0],
        labels=["Negative", "Neutral", "Positive"],
        include_lowest=True,
    )

    urgency_keywords = ["immediately", "unauthorized", "fraud", "urgent", "blocked", "multiple times", "failed"]

    def urgency(text: str, sentiment_score: float) -> str:
        low_text = text.lower()
        keyword_hits = sum(1 for word in urgency_keywords if word in low_text)
        risk = keyword_hits + (1 if sentiment_score < -0.40 else 0)
        if risk >= 3:
            return "High"
        if risk >= 1:
            return "Medium"
        return "Low"

    df["urgency"] = [urgency(t, s) for t, s in zip(df["complaint_text"], df["sentiment_score"])]
    return df


def find_duplicates(df: pd.DataFrame, threshold: float = 0.72) -> pd.DataFrame:
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["complaint_text"])
    sims = cosine_similarity(matrix)

    duplicate_of = []
    duplicate_cluster = []
    cluster_id = 0

    for i in range(len(df)):
        linked = np.where(sims[i, :i] >= threshold)[0]
        if linked.size > 0:
            j = int(linked[0])
            duplicate_of.append(df.iloc[j]["complaint_id"])
            duplicate_cluster.append(f"DUP-{j:04d}")
        else:
            duplicate_of.append("")
            duplicate_cluster.append(f"DUP-{cluster_id:04d}")
            cluster_id += 1

    out = df.copy()
    out["duplicate_of"] = duplicate_of
    out["duplicate_cluster"] = duplicate_cluster
    out["is_duplicate"] = out["duplicate_of"] != ""
    return out


def process_complaints(df: pd.DataFrame) -> Tuple[pd.DataFrame, PipelineArtifacts]:
    artifacts = train_category_model(df)

    out = df.copy()
    out["predicted_category"] = artifacts.classifier.predict(out["complaint_text"])
    out = sentiment_and_urgency(out)
    out = find_duplicates(out)

    # SLA check for demo: high urgency should be addressed within 4h.
    timestamps = pd.to_datetime(out["timestamp"])
    age_hours = (pd.Timestamp.now() - timestamps).dt.total_seconds() / 3600
    out["age_hours"] = age_hours.round(2)
    out["sla_status"] = np.where(
        (out["urgency"] == "High") & (out["age_hours"] > 4),
        "Breached",
        "Within SLA",
    )
    return out, artifacts


def generate_root_cause_insights(df: pd.DataFrame, top_n: int = 3) -> dict:
    """Return simple root-cause style insights: top branches and frequent n-grams per category."""
    from sklearn.feature_extraction.text import CountVectorizer

    insights: dict = {}
    df = df.copy()
    categories = df["predicted_category"].unique()

    for cat in categories:
        sub = df[df["predicted_category"] == cat]
        top_branches = (
            sub["branch"].value_counts().head(top_n).to_dict()
        )

        # find frequent 2-grams in complaint_text for this category
        vect = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
        try:
            X = vect.fit_transform(sub["complaint_text"].fillna(""))
            sums = X.sum(axis=0)
            freqs = [(word, int(sums[0, idx])) for word, idx in vect.vocabulary_.items()]
            freqs_sorted = sorted(freqs, key=lambda x: -x[1])[:top_n]
        except Exception:
            freqs_sorted = []

        insights[cat] = {"top_branches": top_branches, "top_phrases": freqs_sorted}

    return insights
