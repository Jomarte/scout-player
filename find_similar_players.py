"""Find players with similar 2024/25 statistical profiles."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from build_player_embeddings import FEATURE_COLUMNS, FEATURES_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player", help="Player name or unique name fragment")
    parser.add_argument("--top", type=int, default=10, help="Number of matches to show")
    parser.add_argument(
        "--all-positions",
        action="store_true",
        help="Compare against all positions instead of the same position",
    )
    return parser.parse_args()


def select_player(frame: pd.DataFrame, query: str) -> pd.Series:
    matches = frame[
        frame["player_name"].str.contains(query, case=False, na=False, regex=False)
    ]
    if matches.empty:
        raise SystemExit(f"No player found matching: {query}")
    if len(matches) > 1:
        names = ", ".join(matches["player_name"].tolist())
        raise SystemExit(f"Multiple players found: {names}. Use a more specific name.")
    return matches.iloc[0]


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(FEATURES_PATH)
    feature_matrix = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled_features = StandardScaler().fit_transform(feature_matrix)
    target = select_player(frame, args.player)
    target_index = target.name

    candidate_mask = pd.Series(True, index=frame.index)
    candidate_mask &= frame.index != target_index
    if not args.all_positions:
        candidate_mask &= frame["position"] == target["position"]

    distances = np.sqrt(
        np.mean(
            (scaled_features[candidate_mask] - scaled_features[target_index]) ** 2,
            axis=1,
        )
    )
    candidates = frame.loc[candidate_mask, ["player_name", "position", "team_name", "minutes", "cluster"]].copy()
    candidates["distance"] = distances
    candidates["compatibility"] = 100 / (1 + candidates["distance"])
    candidates = candidates.sort_values("distance", ascending=True).head(args.top)
    candidates["distance"] = candidates["distance"].round(3)
    candidates["compatibility"] = candidates["compatibility"].round(2)

    print(f"Target: {target['player_name']} | {target['position']} | {target['team_name']}")
    print("Matches:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
