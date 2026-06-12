from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


SAVANT_VIDEO_BASE = "https://baseballsavant.mlb.com/sporty-videos"
SAVANT_SEARCH_BASE = "https://baseballsavant.mlb.com/statcast_search"
MLB_GAME_FEED_FIELDS = (
    "liveData,plays,allPlays,about,atBatIndex,matchup,batter,id,pitcher,id,"
    "playEvents,pitchNumber,type,isPitch,details,playId"
)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _clean_int(value: object) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def savant_video_url(play_id: str) -> str:
    return f"{SAVANT_VIDEO_BASE}?{urlencode({'playId': play_id})}"


def savant_pitch_search_url(row: pd.Series | dict) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    params = {
        "player_type": "pitcher",
        "game_pk": _clean(getter("game_pk", "")),
        "at_bat_number": _clean(getter("at_bat_number", "")),
        "pitch_number": _clean(getter("pitch_number", "")),
        "game_date_gt": _clean(getter("game_date", "")),
        "game_date_lt": _clean(getter("game_date", "")),
        "pitcher": _clean(getter("pitcher", "")),
        "batter": _clean(getter("batter", "")),
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "type": "details",
    }
    params = {key: value for key, value in params.items() if value}
    return f"{SAVANT_SEARCH_BASE}?{urlencode(params)}"


@lru_cache(maxsize=512)
def game_pitch_play_id_map(game_pk: int) -> dict[tuple[int, int, int | None, int | None], str]:
    """Map one MLB game's pitches to Savant video play IDs.

    Statcast's `at_bat_number` is usually the MLB API's zero-based
    `atBatIndex + 1`. The exact player IDs keep this robust if numbering quirks
    appear in older data.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live?"
        f"{urlencode({'fields': MLB_GAME_FEED_FIELDS})}"
    )
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}

    mapping: dict[tuple[int, int, int | None, int | None], str] = {}
    plays = payload.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play in plays:
        about = play.get("about", {}) or {}
        matchup = play.get("matchup", {}) or {}
        at_bat_index = _clean_int(about.get("atBatIndex"))
        pitcher_id = _clean_int((matchup.get("pitcher") or {}).get("id"))
        batter_id = _clean_int((matchup.get("batter") or {}).get("id"))
        if at_bat_index is None:
            continue

        for event in play.get("playEvents", []):
            play_id = _clean(event.get("playId"))
            pitch_number = _clean_int(event.get("pitchNumber"))
            if not play_id or pitch_number is None or event.get("type") != "pitch":
                continue

            for statcast_ab in {at_bat_index, at_bat_index + 1}:
                mapping[(statcast_ab, pitch_number, pitcher_id, batter_id)] = play_id
            mapping[(at_bat_index + 1, pitch_number, None, None)] = play_id

    return mapping


def _play_id_for_row(row: pd.Series | dict) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    game_pk = _clean_int(getter("game_pk", ""))
    at_bat_number = _clean_int(getter("at_bat_number", ""))
    pitch_number = _clean_int(getter("pitch_number", ""))
    if game_pk is None or at_bat_number is None or pitch_number is None:
        return ""

    mapping = game_pitch_play_id_map(game_pk)
    pitcher_id = _clean_int(getter("pitcher", ""))
    batter_id = _clean_int(getter("batter", ""))
    return mapping.get((at_bat_number, pitch_number, pitcher_id, batter_id), "") or mapping.get(
        (at_bat_number, pitch_number, None, None),
        "",
    )


def build_savant_link(row: pd.Series | dict) -> tuple[str, str]:
    """Return a best-effort Baseball Savant video/search URL and link label."""
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    play_id = _clean(getter("play_id", ""))
    if play_id:
        return savant_video_url(play_id), "video"

    game_pk = _clean(getter("game_pk", ""))
    at_bat_number = _clean(getter("at_bat_number", ""))
    pitch_number = _clean(getter("pitch_number", ""))
    if game_pk and at_bat_number and pitch_number:
        return savant_pitch_search_url(row), "pitch search"

    game_pk = _clean(getter("game_pk", ""))
    if game_pk:
        return f"https://baseballsavant.mlb.com/gamefeed?gamePk={game_pk}", "open on Savant"

    params = {
        "player_type": "pitcher",
        "game_date_gt": _clean(getter("game_date", "")),
        "game_date_lt": _clean(getter("game_date", "")),
        "pitcher": _clean(getter("pitcher", "")),
        "batter": _clean(getter("batter", "")),
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
    }
    params = {key: value for key, value in params.items() if value}
    return f"https://baseballsavant.mlb.com/statcast_search?{urlencode(params)}", "open on Savant"


def add_savant_links(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["savant_url"] = pd.Series(dtype=str)
        out["savant_link_type"] = pd.Series(dtype=str)
        return out
    links = out.apply(build_savant_link, axis=1)
    out["savant_url"] = [item[0] for item in links]
    out["savant_link_type"] = [item[1] for item in links]
    return out


def add_savant_video_links(df: pd.DataFrame, *, max_games: int = 25) -> pd.DataFrame:
    """Resolve direct Savant video URLs for visible pitch rows when practical."""
    out = df.copy()
    if out.empty:
        return add_savant_links(out)
    if "play_id" not in out.columns:
        out["play_id"] = pd.NA

    required = {"game_pk", "at_bat_number", "pitch_number"}
    needs_video = out["play_id"].map(_clean).eq("") if "play_id" in out.columns else pd.Series(True, index=out.index)
    if required.issubset(out.columns) and needs_video.any():
        game_ids = [
            int(game_pk)
            for game_pk in pd.to_numeric(out.loc[needs_video, "game_pk"], errors="coerce").dropna().drop_duplicates()
        ]
        if len(game_ids) <= max_games:
            for idx, row in out.loc[needs_video].iterrows():
                play_id = _play_id_for_row(row)
                if play_id:
                    out.at[idx, "play_id"] = play_id

    return add_savant_links(out)
