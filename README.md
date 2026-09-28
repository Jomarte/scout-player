# Scout Player

Initial data collection for the Premier League player-similarity project.

## Collect data

1. Copy `.env.example` to `.env` and add the API-Football key. Do not commit the key.
2. Set the key in the current PowerShell session:

```powershell
$env:API_FOOTBALL_KEY = "your_api_key"
python .\collect_premier_league.py
```

The collector targets league `39` and the season configured in `collect_premier_league.py` (currently `2023`, corresponding to 2023/24). It saves the raw paginated API responses under `data/raw/` and a flattened CSV under `data/`.

## Test similar players

Install the project dependencies, rebuild the 2024/25 feature space, then search by player name:

```powershell
pip install -r requirements.txt
python .\build_player_embeddings.py
python .\find_similar_players.py "Bukayo Saka"
```

The default search compares players in the same position. Add `--all-positions` to compare the complete player pool.

The dashboard reads `data/premier_league_2024_player_embeddings.csv`. Its selected-player map places each point by distance across the normalized statistical features, so the closest-player ranking and the visual distance use the same metric. Players are tagged with up to two position-agnostic playstyles; two-style players use a gradient marker.
