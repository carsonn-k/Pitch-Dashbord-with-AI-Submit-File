import pandas as pd

from pitch_staff_dashboard.links import add_savant_video_links, build_savant_link


def test_build_savant_link_prefers_direct_video_play_id():
    url, label = build_savant_link({"play_id": "abc-123", "game_pk": 1})

    assert label == "video"
    assert url == "https://baseballsavant.mlb.com/sporty-videos?playId=abc-123"


def test_build_savant_link_uses_pitch_specific_search_before_gamefeed():
    url, label = build_savant_link(
        {
            "game_pk": 633805,
            "at_bat_number": 80,
            "pitch_number": 2,
            "pitcher": 519293,
            "batter": 665862,
            "game_date": "2021-06-01",
        }
    )

    assert label == "pitch search"
    assert url.startswith("https://baseballsavant.mlb.com/statcast_search?")
    assert "game_pk=633805" in url
    assert "at_bat_number=80" in url
    assert "pitch_number=2" in url
    assert "type=details" in url


def test_add_savant_video_links_resolves_visible_pitch_rows(monkeypatch):
    def fake_game_map(game_pk):
        assert game_pk == 633805
        return {
            (80, 2, 519293, 665862): "050195dd-c87b-4749-a7b6-52ae176ed688",
        }

    monkeypatch.setattr("pitch_staff_dashboard.links.game_pitch_play_id_map", fake_game_map)
    df = pd.DataFrame(
        [
            {
                "game_pk": 633805,
                "at_bat_number": 80,
                "pitch_number": 2,
                "pitcher": 519293,
                "batter": 665862,
                "play_id": pd.NA,
            }
        ]
    )

    out = add_savant_video_links(df, max_games=1)

    assert out.loc[0, "savant_link_type"] == "video"
    assert out.loc[0, "savant_url"].endswith("playId=050195dd-c87b-4749-a7b6-52ae176ed688")
