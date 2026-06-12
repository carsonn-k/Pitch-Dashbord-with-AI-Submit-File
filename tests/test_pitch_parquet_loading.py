from __future__ import annotations

import pandas as pd

from pitch_staff_dashboard.pitch_data import (
    load_batter_rows,
    load_pitcher_index,
    load_pitcher_rows,
    pitch_data_row_count,
    write_pitch_parquet,
)


def _pitch_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2025-04-01",
                "game_pk": 1,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 100,
                "batter": 200,
                "player_name": "Pitcher, First",
                "batter_name": "Batter One",
                "balls": 0,
                "strikes": 0,
            },
            {
                "game_date": "2025-04-01",
                "game_pk": 1,
                "at_bat_number": 2,
                "pitch_number": 1,
                "pitcher": 100,
                "batter": 201,
                "player_name": "Pitcher, First",
                "batter_name": "Batter Two",
                "balls": 1,
                "strikes": 0,
            },
            {
                "game_date": "2025-04-02",
                "game_pk": 2,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 101,
                "batter": 200,
                "player_name": "Pitcher, Second",
                "batter_name": "Batter One",
                "balls": 0,
                "strikes": 1,
            },
        ]
    )


def test_parquet_pitcher_index_reads_minimal_dataset(tmp_path):
    path = tmp_path / "pitches.parquet"
    write_pitch_parquet(_pitch_rows(), path)

    index = load_pitcher_index(path)

    assert index[["pitcher", "Pitcher", "Pitches"]].to_dict("records") == [
        {"pitcher": 100, "Pitcher": "First Pitcher", "Pitches": 2},
        {"pitcher": 101, "Pitcher": "Second Pitcher", "Pitches": 1},
    ]


def test_parquet_pitcher_filter_returns_matching_rows_and_total_count(tmp_path):
    path = tmp_path / "pitches.parquet"
    write_pitch_parquet(_pitch_rows(), path)

    rows, total_rows = load_pitcher_rows(path, 100)

    assert total_rows == 3
    assert rows["pitcher"].dropna().astype(int).unique().tolist() == [100]
    assert len(rows) == 2


def test_parquet_batter_filter_returns_matching_rows_and_total_count(tmp_path):
    path = tmp_path / "pitches.parquet"
    write_pitch_parquet(_pitch_rows(), path)

    rows, total_rows = load_batter_rows(path, 200)

    assert total_rows == 3
    assert rows["batter"].dropna().astype(int).unique().tolist() == [200]
    assert len(rows) == 2
    assert pitch_data_row_count(path) == 3
