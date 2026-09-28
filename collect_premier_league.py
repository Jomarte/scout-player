"""Collect Premier League player statistics from API-Football."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39
SEASON = 2023
PER_PAGE = 20
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/premier_league_2023_player_stats.csv")


def load_api_key() -> str:
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "API_FOOTBALL_KEY":
                    api_key = value.strip().strip('"').strip("'")
                    break
    if not api_key:
        raise SystemExit(
            "Missing API_FOOTBALL_KEY. Set it in the environment before running."
        )
    return api_key


def request_json(api_key: str, endpoint: str, params: dict[str, int]) -> dict:
    query = urlencode(params)
    request = Request(
        f"{BASE_URL}/{endpoint}?{query}",
        headers={"x-apisports-key": api_key, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise SystemExit(f"API request failed with HTTP {error.code}.") from error
    except URLError as error:
        raise SystemExit(f"Could not reach API-Football: {error.reason}") from error


def save_raw_response(payload: dict, filename: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_raw_response(filename: str) -> dict | None:
    output_path = RAW_DIR / filename
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))


def flatten_player(player_entry: dict) -> list[dict]:
    player = player_entry.get("player", {})
    rows = []
    for statistic in player_entry.get("statistics", []):
        league = statistic.get("league") or {}
        if league.get("id") != LEAGUE_ID:
            continue
        games = statistic.get("games") or {}
        minutes = games.get("minutes") or 0
        if minutes <= 0:
            continue
        goals = statistic.get("goals") or {}
        shots = statistic.get("shots") or {}
        passes = statistic.get("passes") or {}
        tackles = statistic.get("tackles") or {}
        duels = statistic.get("duels") or {}
        dribbles = statistic.get("dribbles") or {}
        fouls = statistic.get("fouls") or {}
        cards = statistic.get("cards") or {}

        rows.append(
            {
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "first_name": player.get("firstname"),
                "last_name": player.get("lastname"),
                "age": player.get("age"),
                "nationality": player.get("nationality"),
                "team_id": (statistic.get("team") or {}).get("id"),
                "team_name": (statistic.get("team") or {}).get("name"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "position": games.get("position"),
                "appearances": games.get("appearences"),
                "minutes": minutes,
                "lineups": games.get("lineups"),
                "rating": games.get("rating"),
                "goals": goals.get("total"),
                "assists": goals.get("assists"),
                "shots_total": shots.get("total"),
                "shots_on": shots.get("on"),
                "passes_total": passes.get("total"),
                "passes_key": passes.get("key"),
                "passes_accuracy": passes.get("accuracy"),
                "tackles_total": tackles.get("total"),
                "tackles_blocks": tackles.get("blocks"),
                "tackles_interceptions": tackles.get("interceptions"),
                "duels_total": duels.get("total"),
                "duels_won": duels.get("won"),
                "dribbles_attempts": dribbles.get("attempts"),
                "dribbles_success": dribbles.get("success"),
                "dribbles_past": dribbles.get("past"),
                "fouls_drawn": fouls.get("drawn"),
                "fouls_committed": fouls.get("committed"),
                "yellow_cards": cards.get("yellow"),
                "red_cards": cards.get("red"),
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("The API returned no player statistics.")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    api_key = load_api_key()
    teams_filename = f"teams_league_{LEAGUE_ID}_season_{SEASON}.json"
    teams_payload = load_raw_response(teams_filename)
    if teams_payload is None:
        teams_payload = request_json(
            api_key,
            "teams",
            {"league": LEAGUE_ID, "season": SEASON},
        )
    errors = teams_payload.get("errors") or {}
    if errors:
        raise SystemExit(f"API returned errors: {errors}")

    if not (RAW_DIR / teams_filename).exists():
        save_raw_response(teams_payload, teams_filename)
    teams = teams_payload.get("response", [])
    if not teams:
        raise SystemExit("The API returned no Premier League teams.")

    all_rows = []
    for team_number, team_entry in enumerate(teams, start=1):
        team = team_entry.get("team") or {}
        team_id = team.get("id")
        team_name = team.get("name") or f"team-{team_id}"
        if not team_id:
            continue

        first_filename = f"players_team_{team_id}_season_{SEASON}_page_1.json"
        first_page = load_raw_response(first_filename)
        if first_page is None:
            try:
                first_page = request_json(
                    api_key,
                    "players",
                    {"team": team_id, "season": SEASON, "page": 1},
                )
            except SystemExit as error:
                if "HTTP 429" in str(error):
                    print("Rate limit reached. Run the collector again later to resume.")
                    break
                raise
        errors = first_page.get("errors") or {}
        if errors:
            raise SystemExit(f"API returned errors for {team_name}: {errors}")

        total_pages = min(int((first_page.get("paging") or {}).get("total") or 1), 3)
        for page in range(1, total_pages + 1):
            filename = f"players_team_{team_id}_season_{SEASON}_page_{page}.json"
            payload = load_raw_response(filename)
            if payload is None:
                payload = (
                    first_page
                    if page == 1
                    else None
                )
                if payload is None:
                    try:
                        payload = request_json(
                            api_key,
                            "players",
                            {"team": team_id, "season": SEASON, "page": page},
                        )
                    except SystemExit as error:
                        if "HTTP 429" in str(error):
                            print("Rate limit reached. Run the collector again later to resume.")
                            break
                        raise
            page_errors = payload.get("errors") or {}
            if page_errors:
                raise SystemExit(f"API returned errors for {team_name}: {page_errors}")
            if not (RAW_DIR / filename).exists():
                save_raw_response(payload, filename)
            for player_entry in payload.get("response", []):
                all_rows.extend(flatten_player(player_entry))
        print(f"Collected {team_number}/{len(teams)}: {team_name}")

    write_csv(all_rows)
    print(f"Saved {len(all_rows)} player-team rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
