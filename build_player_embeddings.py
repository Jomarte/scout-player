"""Build per-90 features, K-Means clusters, and a t-SNE player map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/premier_league_2024_player_stats.csv")
FEATURES_PATH = Path("data/premier_league_2024_player_features.csv")
EMBEDDINGS_PATH = Path("data/premier_league_2024_player_embeddings.csv")
MIN_MINUTES = 300
N_CLUSTERS = 6

COUNT_FEATURES = [
    "goals",
    "assists",
    "shots_total",
    "shots_on",
    "passes_total",
    "passes_key",
    "tackles_total",
    "tackles_blocks",
    "tackles_interceptions",
    "duels_total",
    "duels_won",
    "dribbles_attempts",
    "dribbles_success",
    "dribbles_past",
    "fouls_drawn",
    "fouls_committed",
    "yellow_cards",
    "red_cards",
]
FEATURE_COLUMNS = [f"{column}_per90" for column in COUNT_FEATURES] + [
    "passes_accuracy",
    "duels_won_pct",
    "dribbles_success_pct",
]


def load_and_aggregate() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH)
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    for column in COUNT_FEATURES + ["passes_accuracy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    aggregations = {column: "sum" for column in COUNT_FEATURES + ["minutes"]}
    aggregations.update(
        {
            "player_name": "first",
            "position": "first",
            "team_name": lambda values: " / ".join(sorted(set(values.dropna()))),
            "passes_accuracy": "mean",
        }
    )
    aggregated = frame.groupby("player_id", as_index=False).agg(aggregations)
    aggregated = aggregated[aggregated["minutes"] >= MIN_MINUTES].copy()
    return aggregated


def build_features(players: pd.DataFrame) -> pd.DataFrame:
    features = players[["player_id", "player_name", "position", "team_name", "minutes"]].copy()
    for column in COUNT_FEATURES:
        features[f"{column}_per90"] = np.round(
            players[column] * 90 / players["minutes"], 3
        )

    features["passes_accuracy"] = np.round(players["passes_accuracy"], 3)
    features["duels_won_pct"] = np.round(
        np.divide(
            players["duels_won"],
            players["duels_total"],
            out=np.zeros(len(players), dtype=float),
            where=players["duels_total"].to_numpy() != 0,
        )
        * 100,
        3,
    )
    features["dribbles_success_pct"] = np.round(
        np.divide(
            players["dribbles_success"],
            players["dribbles_attempts"],
            out=np.zeros(len(players), dtype=float),
            where=players["dribbles_attempts"].to_numpy() != 0,
        )
        * 100,
        3,
    )
    return features


def main() -> None:
    players = load_and_aggregate()
    features = build_features(players)
    feature_matrix = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    cluster_count = min(N_CLUSTERS, len(features))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
    features["cluster"] = model.fit_predict(scaled_features)

    perplexity = min(30, max(5, (len(features) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200.0,
        init="pca",
        metric="euclidean",
        random_state=42,
    ).fit_transform(scaled_features)
    features["embedding_x"] = np.round(embedding[:, 0], 4)
    features["embedding_y"] = np.round(embedding[:, 1], 4)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    features.to_csv(EMBEDDINGS_PATH, index=False, encoding="utf-8-sig")
    print(f"Players used: {len(features)}")
    print(f"Features per player: {len(FEATURE_COLUMNS)}")
    print(f"Clusters: {cluster_count}")
    print(f"t-SNE perplexity: {perplexity}")
    print(f"Saved: {FEATURES_PATH}")
    print(f"Saved: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
