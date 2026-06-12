from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path(os.environ.get("STATCAST_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))
DEFAULT_SEASON_DIR = Path(os.environ.get("STATCAST_SEASON_DIR", PROJECT_ROOT / "data" / "seasons"))
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_statcast.parquet"


NUMERIC_COLUMNS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "inning",
    "balls",
    "strikes",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "on_1b",
    "on_2b",
    "on_3b",
    "outs_when_up",
    "sz_top",
    "sz_bot",
    "game_year",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]


TEXT_COLUMNS = [
    "game_date",
    "player_name",
    "pitcher_name",
    "batter_name",
    "stand",
    "p_throws",
    "pitch_type",
    "pitch_name",
    "description",
    "events",
    "inning_topbot",
    "home_team",
    "away_team",
    "batting_team",
    "fielding_team",
    "bb_type",
    "play_id",
    "sv_id",
]


def iter_date_chunks(start_dt: str | date, end_dt: str | date, chunk_days: int = 7) -> Iterable[tuple[str, str]]:
    """Yield inclusive date chunks for Baseball Savant pulls."""
    start = pd.to_datetime(start_dt).date()
    end = pd.to_datetime(end_dt).date()
    step = max(int(chunk_days), 1)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=step - 1), end)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def _write_frame(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_sample_data() -> pd.DataFrame:
    """Load the included tiny Statcast-like dataset for offline app testing."""
    return normalize_statcast_frame(pd.read_parquet(SAMPLE_DATA_PATH))


def season_date_range(season: int, today: date | None = None) -> tuple[date, date]:
    """Return a practical MLB Statcast season window for one local season file."""
    current = today or date.today()
    start = date(int(season), 3, 1)
    scheduled_end = date(int(season), 11, 30)
    if int(season) == current.year:
        return start, min(scheduled_end, current)
    return start, scheduled_end


def season_dataset_path(season: int, season_dir: Path = DEFAULT_SEASON_DIR) -> Path:
    return Path(season_dir) / f"statcast_{int(season)}.parquet"


def _read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"No season dataset found at {path}.")


def season_dataset_exists(season: int, season_dir: Path = DEFAULT_SEASON_DIR) -> bool:
    path = season_dataset_path(season, season_dir)
    return path.exists()


def load_season_dataset(season: int, season_dir: Path = DEFAULT_SEASON_DIR) -> pd.DataFrame:
    """Load one local full-season Statcast file."""
    return normalize_statcast_frame(_read_frame(season_dataset_path(season, season_dir)))


def fetch_statcast_range(start_dt: str | date, end_dt: str | date, *, chunk_days: int = 7) -> pd.DataFrame:
    """Fetch Statcast in pull batches without writing per-batch cache files."""
    chunks: list[pd.DataFrame] = []

    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError("pybaseball is not installed. Run `pip install -r requirements.txt`.") from exc

    for chunk_start, chunk_end in iter_date_chunks(start_dt, end_dt, chunk_days=chunk_days):
        pulled = statcast(start_dt=chunk_start, end_dt=chunk_end)
        if pulled is not None and not pulled.empty:
            chunks.append(pulled)

    if not chunks:
        return normalize_statcast_frame(pd.DataFrame())
    return normalize_statcast_frame(pd.concat(chunks, ignore_index=True))


def build_season_dataset(
    season: int,
    *,
    season_dir: Path = DEFAULT_SEASON_DIR,
    through_date: date | None = None,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Fetch one season, save it as a single local dataset, and return it.

    The full-season file is the app's working dataset. Internally the fetch is
    still batched by date so Baseball Savant requests stay reasonably sized.
    """
    start_dt, default_end = season_date_range(season)
    end_dt = min(through_date, default_end) if through_date else default_end
    df = fetch_statcast_range(start_dt, end_dt, chunk_days=chunk_days)
    _write_frame(df, season_dataset_path(season, season_dir))
    return df


def list_season_datasets(season_dir: Path = DEFAULT_SEASON_DIR) -> list[Path]:
    path = Path(season_dir)
    if not path.exists():
        return []
    return sorted(path.glob("statcast_*.parquet"))


def normalize_display_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    if "," in text:
        last, first = [piece.strip() for piece in text.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return text


def _derive_team(row: pd.Series, top_value: str, bottom_value: str) -> object:
    topbot = str(row.get("inning_topbot", "")).lower()
    if topbot.startswith("top"):
        return row.get(top_value)
    if topbot.startswith("bot"):
        return row.get(bottom_value)
    return pd.NA


def normalize_statcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Statcast columns used by the dashboard without assuming all exist."""
    normalized = df.copy()

    for col in NUMERIC_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in TEXT_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.date

    if normalized["game_year"].isna().all() and "game_date" in normalized.columns:
        normalized["game_year"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.year

    if normalized["batting_team"].isna().all() and {"home_team", "away_team", "inning_topbot"}.issubset(normalized.columns):
        normalized["batting_team"] = normalized.apply(lambda row: _derive_team(row, "away_team", "home_team"), axis=1)

    if normalized["fielding_team"].isna().all() and {"home_team", "away_team", "inning_topbot"}.issubset(normalized.columns):
        normalized["fielding_team"] = normalized.apply(lambda row: _derive_team(row, "home_team", "away_team"), axis=1)

    if normalized["pitcher_name"].isna().all() and "player_name" in normalized.columns:
        normalized["pitcher_name"] = normalized["player_name"]

    normalized["pitcher_name"] = normalized["pitcher_name"].map(normalize_display_name)
    normalized["batter_name"] = normalized["batter_name"].map(normalize_display_name)

    normalized["pitcher_display"] = normalized.apply(
        lambda row: row["pitcher_name"] if row["pitcher_name"] else f"Pitcher {int(row['pitcher'])}" if pd.notna(row["pitcher"]) else "Unknown Pitcher",
        axis=1,
    )
    normalized["batter_display"] = normalized.apply(
        lambda row: row["batter_name"] if row["batter_name"] else f"Batter {int(row['batter'])}" if pd.notna(row["batter"]) else "Unknown Batter",
        axis=1,
    )

    for col in ["home_team", "away_team", "batting_team", "fielding_team", "stand", "p_throws", "pitch_type"]:
        normalized[col] = normalized[col].astype("string").str.strip()

    normalized["count"] = normalized.apply(
        lambda row: f"{int(row['balls'])}-{int(row['strikes'])}" if pd.notna(row["balls"]) and pd.notna(row["strikes"]) else "",
        axis=1,
    )

    return normalized


def attach_player_names(
    df: pd.DataFrame,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fill batter/pitcher names from MLBAM IDs when the raw Statcast pull lacks names."""
    out = normalize_statcast_frame(df)
    ids = pd.concat([out["pitcher"], out["batter"]], ignore_index=True).dropna().astype(int).unique().tolist()
    if not ids:
        return out

    cache_path = Path(cache_dir) / "player_lookup.parquet"
    lookup = pd.DataFrame()
    if cache_path.exists() and not force_refresh:
        lookup = pd.read_parquet(cache_path)

    missing_ids = set(ids)
    if not lookup.empty and "key_mlbam" in lookup.columns:
        missing_ids -= set(pd.to_numeric(lookup["key_mlbam"], errors="coerce").dropna().astype(int))

    if missing_ids:
        try:
            from pybaseball import playerid_reverse_lookup

            fetched = playerid_reverse_lookup(sorted(missing_ids), key_type="mlbam")
            lookup = pd.concat([lookup, fetched], ignore_index=True).drop_duplicates("key_mlbam", keep="last")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lookup.to_parquet(cache_path, index=False)
        except Exception:
            return out

    if lookup.empty or "key_mlbam" not in lookup.columns:
        return out

    lookup = lookup.copy()
    lookup["key_mlbam"] = pd.to_numeric(lookup["key_mlbam"], errors="coerce")
    lookup["display_name"] = (
        lookup.get("name_first", pd.Series("", index=lookup.index)).fillna("").astype(str).str.strip()
        + " "
        + lookup.get("name_last", pd.Series("", index=lookup.index)).fillna("").astype(str).str.strip()
    ).str.strip()
    name_map = lookup.dropna(subset=["key_mlbam"]).set_index("key_mlbam")["display_name"].to_dict()

    missing_pitcher = out["pitcher_name"].eq("")
    missing_batter = out["batter_name"].eq("")
    out.loc[missing_pitcher, "pitcher_name"] = out.loc[missing_pitcher, "pitcher"].map(name_map).fillna("")
    out.loc[missing_batter, "batter_name"] = out.loc[missing_batter, "batter"].map(name_map).fillna("")
    return normalize_statcast_frame(out)
