import pandas as pd

from pitch_staff_dashboard.transforms import prepare_pitch_sequences, validate_pitch_order


def _base_rows():
    return [
        {
            "game_pk": 1,
            "at_bat_number": 10,
            "pitcher": 100,
            "batter": 200,
            "inning": 1,
            "inning_topbot": "Top",
            "pitch_number": 3,
            "description": "hit_into_play",
            "events": "single",
            "balls": 1,
            "strikes": 1,
        },
        {
            "game_pk": 1,
            "at_bat_number": 10,
            "pitcher": 100,
            "batter": 200,
            "inning": 1,
            "inning_topbot": "Top",
            "pitch_number": 1,
            "description": "called_strike",
            "events": None,
            "balls": 0,
            "strikes": 0,
        },
        {
            "game_pk": 1,
            "at_bat_number": 10,
            "pitcher": 100,
            "batter": 200,
            "inning": 1,
            "inning_topbot": "Top",
            "pitch_number": 2,
            "description": "ball",
            "events": None,
            "balls": 0,
            "strikes": 1,
        },
    ]


def test_prepare_pitch_sequences_sorts_inside_plate_appearance():
    df = pd.DataFrame(_base_rows())

    sequenced = prepare_pitch_sequences(df)

    assert sequenced["pitch_number"].tolist() == [1, 2, 3]
    assert sequenced["pa_pitch_index"].tolist() == [1, 2, 3]
    assert sequenced["pa_result"].tolist() == ["single", "single", "single"]


def test_plate_appearance_key_keeps_batter_matchups_separate():
    rows = _base_rows()
    rows.append(
        {
            "game_pk": 1,
            "at_bat_number": 10,
            "pitcher": 100,
            "batter": 201,
            "inning": 1,
            "inning_topbot": "Top",
            "pitch_number": 1,
            "description": "called_strike",
            "events": None,
            "balls": 0,
            "strikes": 0,
        }
    )
    sequenced = prepare_pitch_sequences(pd.DataFrame(rows))

    assert sequenced["pa_key"].nunique() == 2
    assert sequenced.groupby("pa_key")["batter"].nunique().max() == 1


def test_validate_pitch_order_detects_duplicate_pitch_number():
    rows = _base_rows()
    rows[1]["pitch_number"] = 1
    rows[2]["pitch_number"] = 1
    df = pd.DataFrame(rows)

    issues = validate_pitch_order(prepare_pitch_sequences(df))

    assert not issues.empty
    assert issues.iloc[0]["reason"] == "pitch_number not strictly increasing"


def test_validate_pitch_order_detects_missing_pitch_number():
    rows = _base_rows()
    rows[1]["pitch_number"] = None

    issues = validate_pitch_order(pd.DataFrame(rows))

    assert not issues.empty
    assert issues.iloc[0]["reason"] == "missing pitch_number"
