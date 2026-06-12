from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .data import NUMERIC_COLUMNS, PROJECT_ROOT, iter_date_chunks, normalize_display_name, normalize_statcast_frame


ALL_PITCHES_LAST2_PARQUET_PATH = PROJECT_ROOT / "data" / "statcast_all_pitches_last2years.parquet"
ALL_PITCHES_LAST5_PARQUET_PATH = PROJECT_ROOT / "data" / "statcast_all_pitches_last5years.parquet"
PITCH_DEDUP_COLUMNS = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"]
PITCH_INTEGER_COLUMNS = {
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "inning",
    "balls",
    "strikes",
    "zone",
    "on_1b",
    "on_2b",
    "on_3b",
    "outs_when_up",
    "game_year",
}


MLB_TEAMS: dict[str, int] = {
    "Arizona Diamondbacks": 109,
    "Athletics": 133,
    "Atlanta Braves": 144,
    "Baltimore Orioles": 110,
    "Boston Red Sox": 111,
    "Chicago Cubs": 112,
    "Chicago White Sox": 145,
    "Cincinnati Reds": 113,
    "Cleveland Guardians": 114,
    "Colorado Rockies": 115,
    "Detroit Tigers": 116,
    "Houston Astros": 117,
    "Kansas City Royals": 118,
    "Los Angeles Angels": 108,
    "Los Angeles Dodgers": 119,
    "Miami Marlins": 146,
    "Milwaukee Brewers": 158,
    "Minnesota Twins": 142,
    "New York Mets": 121,
    "New York Yankees": 147,
    "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134,
    "San Diego Padres": 135,
    "San Francisco Giants": 137,
    "Seattle Mariners": 136,
    "St. Louis Cardinals": 138,
    "Tampa Bay Rays": 139,
    "Texas Rangers": 140,
    "Toronto Blue Jays": 141,
    "Washington Nationals": 120,
}


def last_five_year_window(today: date | None = None) -> tuple[date, date]:
    end = today or date.today()
    try:
        start = end.replace(year=end.year - 5)
    except ValueError:
        start = end.replace(year=end.year - 5, day=28)
    return start, end


def rolling_year_window(years: int, today: date | None = None) -> tuple[date, date]:
    end = today or date.today()
    try:
        start = end.replace(year=end.year - int(years))
    except ValueError:
        start = end.replace(year=end.year - int(years), day=28)
    return start, end


def fetch_all_pitches(start_dt: date, end_dt: date, *, chunk_days: int = 7) -> pd.DataFrame:
    """Fetch all MLB pitch-level Statcast rows in bounded date chunks."""
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError("pybaseball is not installed. Run `pip install -r requirements.txt`.") from exc

    chunks: list[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_date_chunks(start_dt, end_dt, chunk_days=chunk_days):
        try:
            pulled = statcast(start_dt=chunk_start, end_dt=chunk_end, parallel=False)
        except TypeError:
            pulled = statcast(start_dt=chunk_start, end_dt=chunk_end)
        if pulled is not None and not pulled.empty:
            chunks.append(pulled)

    if not chunks:
        return normalize_statcast_frame(pd.DataFrame())
    return normalize_statcast_frame(pd.concat(chunks, ignore_index=True))


def is_parquet_dataset(path: Path) -> bool:
    return path.suffix == ".parquet" or (path.exists() and path.is_dir())


def resolve_pitch_data_path(path: Path) -> Path:
    return path


def prepare_pitch_storage_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Parquet storage types predictable across fetched and converted data."""
    prepared = df.copy()
    for col in prepared.columns:
        if col in PITCH_INTEGER_COLUMNS:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype("Int64")
        elif col in NUMERIC_COLUMNS:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
        elif col == "game_date":
            prepared[col] = pd.to_datetime(prepared[col], errors="coerce")
        else:
            prepared[col] = prepared[col].where(pd.notna(prepared[col]), None).astype("string")
    return prepared


def write_pitch_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    storage_df = prepare_pitch_storage_frame(df)
    partial_path = path.with_suffix(path.suffix + ".partial")
    try:
        storage_df.to_parquet(partial_path, index=False, compression="zstd")
    except Exception:
        storage_df.to_parquet(partial_path, index=False)
    partial_path.replace(path)
    return path


def pitch_data_row_count(path: Path) -> int:
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        return 0
    if resolved.is_file():
        return pq.ParquetFile(resolved).metadata.num_rows
    return int(ds.dataset(str(resolved), format="parquet").count_rows())


def pitch_data_size_bytes(path: Path) -> int:
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        return 0
    if resolved.is_dir():
        return sum(child.stat().st_size for child in resolved.rglob("*") if child.is_file())
    return resolved.stat().st_size


def pitch_data_columns(path: Path) -> list[str]:
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pitch dataset not found: {path}")
    if resolved.is_file():
        return list(pq.ParquetFile(resolved).schema_arrow.names)
    return list(ds.dataset(str(resolved), format="parquet").schema.names)


def pitch_data_date_range(path: Path) -> tuple[date | None, date | None]:
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists() or "game_date" not in pitch_data_columns(resolved):
        return None, None

    if resolved.is_file():
        parquet_file = pq.ParquetFile(resolved)
        date_index = parquet_file.schema_arrow.names.index("game_date")
        mins: list[object] = []
        maxes: list[object] = []
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            stats = parquet_file.metadata.row_group(row_group_index).column(date_index).statistics
            if stats is not None and stats.has_min_max:
                mins.append(stats.min)
                maxes.append(stats.max)
        if mins and maxes:
            min_date = pd.to_datetime(pd.Series(mins), errors="coerce").dropna().min()
            max_date = pd.to_datetime(pd.Series(maxes), errors="coerce").dropna().max()
            if pd.notna(min_date) and pd.notna(max_date):
                return min_date.date(), max_date.date()

    dates = pd.to_datetime(pd.read_parquet(resolved, columns=["game_date"])["game_date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date(), dates.max().date()


def _read_parquet_rows(path: Path, filter_col: str, filter_value: int) -> tuple[pd.DataFrame, int]:
    total_rows = pitch_data_row_count(path)
    columns = pitch_data_columns(path)
    if filter_col not in columns:
        raise ValueError(f"`{filter_col}` column is missing from {path}")
    df = pd.read_parquet(path, filters=[(filter_col, "==", int(filter_value))])
    return normalize_statcast_frame(df), total_rows


def _pitcher_index_from_frame(df: pd.DataFrame, name_col: str | None) -> pd.DataFrame:
    if "pitcher" not in df.columns:
        raise ValueError("`pitcher` column is missing from pitch dataset")

    pitcher_values = pd.to_numeric(df["pitcher"], errors="coerce")
    valid = df[pitcher_values.notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=["pitcher", "Pitcher", "Pitches"])

    valid["pitcher"] = pitcher_values.loc[valid.index].astype(int)
    counts = valid.groupby("pitcher").size().to_dict()

    names: dict[int, str] = {}
    if name_col and name_col in valid.columns:
        name_rows = valid[["pitcher", name_col]].dropna().drop_duplicates("pitcher")
        for row in name_rows.itertuples(index=False):
            name = normalize_display_name(getattr(row, name_col))
            if name and int(row.pitcher) not in names:
                names[int(row.pitcher)] = name

    rows = [
        {
            "pitcher": pitcher_id,
            "Pitcher": names.get(pitcher_id, f"Pitcher {pitcher_id}"),
            "Pitches": int(pitch_count),
        }
        for pitcher_id, pitch_count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["Pitcher", "pitcher"]).reset_index(drop=True)


def build_all_pitches_parquet(
    output_path: Path = ALL_PITCHES_LAST5_PARQUET_PATH,
    *,
    start_dt: date | None = None,
    end_dt: date | None = None,
    chunk_days: int = 7,
    rolling_years: int = 5,
) -> pd.DataFrame:
    """Build the local all-pitches Parquet dataset."""
    default_start, default_end = rolling_year_window(rolling_years)
    start = start_dt or default_start
    end = end_dt or default_end
    pitches = fetch_all_pitches(start, end, chunk_days=chunk_days)
    write_pitch_parquet(pitches, output_path)
    return pitches


def load_pitch_data(path: Path) -> pd.DataFrame:
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pitch dataset not found: {path}")
    return normalize_statcast_frame(pd.read_parquet(resolved))


def load_pitcher_rows(path: Path, pitcher_id: int, *, chunksize: int = 100_000) -> tuple[pd.DataFrame, int]:
    """Load only one pitcher's rows from a Parquet-first pitch dataset."""
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pitch dataset not found: {path}")

    return _read_parquet_rows(resolved, "pitcher", pitcher_id)


def load_batter_rows(path: Path, batter_id: int, *, chunksize: int = 100_000) -> tuple[pd.DataFrame, int]:
    """Load only one batter's rows from a Parquet-first pitch dataset."""
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pitch dataset not found: {path}")

    return _read_parquet_rows(resolved, "batter", batter_id)


def load_pitcher_index(path: Path, *, chunksize: int = 100_000) -> pd.DataFrame:
    """Build a selectable pitcher list while reading the smallest useful slice."""
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pitch dataset not found: {path}")

    columns = pitch_data_columns(resolved)
    name_col = "pitcher_name" if "pitcher_name" in columns else "player_name" if "player_name" in columns else None
    usecols = ["pitcher"] + ([name_col] if name_col else [])

    return _pitcher_index_from_frame(pd.read_parquet(resolved, columns=usecols), name_col)


def _dedupe_pitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in PITCH_DEDUP_COLUMNS if col in df.columns]
    if keys:
        return df.drop_duplicates(keys, keep="last").reset_index(drop=True)
    return df.drop_duplicates(keep="last").reset_index(drop=True)


def _trim_to_window(df: pd.DataFrame, start: date) -> pd.DataFrame:
    if df.empty or "game_date" not in df.columns:
        return df
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    return df[dates.dt.date.ge(start)].copy()


def refresh_pitch_parquet(
    path: Path,
    fetcher: Callable[[date, date], pd.DataFrame],
    *,
    today: date | None = None,
    rolling_years: int = 5,
    overlap_days: int = 0,
) -> dict[str, object]:
    """Fetch recent rows since the local Parquet dataset's latest game date."""
    current = today or date.today()
    window_start, _ = rolling_year_window(rolling_years, current)
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        return {"status": "missing", "rows_added": 0, "start": None, "end": None}

    _, max_date = pitch_data_date_range(resolved)
    if max_date is None:
        refresh_start = window_start
    else:
        refresh_start = max(window_start, max_date + timedelta(days=1) - timedelta(days=max(overlap_days, 0)))

    if refresh_start > current:
        return {"status": "up_to_date", "rows_added": 0, "start": refresh_start, "end": current}

    fetched = fetcher(refresh_start, current)
    if fetched.empty:
        return {"status": "no_new_rows", "rows_added": 0, "start": refresh_start, "end": current}

    existing = pd.read_parquet(resolved)
    before_rows = len(existing)
    combined = pd.concat([existing, fetched], ignore_index=True)
    combined = prepare_pitch_storage_frame(_trim_to_window(_dedupe_pitch_rows(combined), window_start))
    write_pitch_parquet(combined, resolved)

    return {
        "status": "updated",
        "rows_added": max(len(combined) - before_rows, 0),
        "rows_total": len(combined),
        "start": refresh_start,
        "end": current,
    }


def pitch_refresh_status(path: Path, *, today: date | None = None, rolling_years: int = 5) -> dict[str, object]:
    current = today or date.today()
    window_start, _ = rolling_year_window(rolling_years, current)
    resolved = resolve_pitch_data_path(path)
    if not resolved.exists():
        return {"status": "missing", "min_date": None, "max_date": None, "missing_days": None}

    min_date, max_date = pitch_data_date_range(resolved)
    if max_date is None:
        return {"status": "unknown", "min_date": min_date, "max_date": None, "missing_days": None}

    missing_days = max((current - max_date).days, 0)
    if missing_days == 0:
        status = "up_to_date"
    else:
        status = "stale"
    return {
        "status": status,
        "min_date": min_date,
        "max_date": max_date,
        "missing_days": missing_days,
        "window_start": window_start,
    }


def refresh_all_pitches_parquet(
    path: Path = ALL_PITCHES_LAST5_PARQUET_PATH,
    *,
    today: date | None = None,
    rolling_years: int = 5,
) -> dict[str, object]:
    return refresh_pitch_parquet(path, lambda start, end: fetch_all_pitches(start, end), today=today, rolling_years=rolling_years)


def fetch_current_roster(team_id: int, roster_type: str = "active", timeout: int = 20) -> pd.DataFrame:
    """Fetch the current MLB roster from the public MLB Stats API."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType={roster_type}&hydrate=person"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Could not fetch current roster from MLB Stats API: {exc}") from exc

    rows = []
    for entry in payload.get("roster", []):
        person = entry.get("person", {}) or {}
        position = entry.get("position", {}) or {}
        status = entry.get("status", {}) or {}
        rows.append(
            {
                "batter": person.get("id"),
                "Current Player": person.get("fullName", ""),
                "Position": position.get("abbreviation", ""),
                "Position Type": position.get("type", ""),
                "Status": status.get("description", ""),
                "Bats": (person.get("batSide") or {}).get("code", ""),
                "Throws": (person.get("pitchHand") or {}).get("code", ""),
                "Jersey": entry.get("jerseyNumber", ""),
            }
        )
    roster = pd.DataFrame(rows)
    if roster.empty:
        return roster
    roster["batter"] = pd.to_numeric(roster["batter"], errors="coerce")
    return roster.dropna(subset=["batter"]).sort_values(["Position Type", "Current Player"]).reset_index(drop=True)


def hitters_from_roster(roster: pd.DataFrame, *, include_pitchers: bool = False) -> pd.DataFrame:
    if roster.empty or include_pitchers:
        return roster.copy()
    position = roster.get("Position", pd.Series("", index=roster.index)).fillna("").astype(str)
    position_type = roster.get("Position Type", pd.Series("", index=roster.index)).fillna("").astype(str)
    return roster[~position.eq("P") & ~position_type.eq("Pitcher")].copy()


def filter_to_roster_matchups(pitches: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty or roster.empty:
        return pitches.iloc[0:0].copy()
    roster_ids = set(pd.to_numeric(roster["batter"], errors="coerce").dropna().astype(int))
    matchups = pitches[pd.to_numeric(pitches["batter"], errors="coerce").isin(roster_ids)].copy()
    name_map = roster.set_index("batter")["Current Player"].to_dict()
    position_map = roster.set_index("batter")["Position"].to_dict()
    matchups["Current Player"] = matchups["batter"].map(name_map).fillna(matchups.get("batter_display", ""))
    matchups["Current Position"] = matchups["batter"].map(position_map)
    matchups["batter_display"] = matchups["Current Player"].combine_first(matchups.get("batter_display", ""))
    return matchups


def roster_matchup_table(pitches: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if roster.empty:
        return pd.DataFrame()

    prepared = pitches.copy()
    prepared["batter_int"] = pd.to_numeric(prepared.get("batter"), errors="coerce")

    for _, player in roster.iterrows():
        batter_id = int(player["batter"])
        group = prepared[prepared["batter_int"].eq(batter_id)]
        events = (
            group.dropna(subset=["events"])["events"].astype(str).replace("", pd.NA).dropna().unique().tolist()
            if not group.empty and "events" in group
            else []
        )
        rows.append(
            {
                "Current Player": player.get("Current Player", ""),
                "Position": player.get("Position", ""),
                "Bats": player.get("Bats", ""),
                "Batter ID": batter_id,
                "PA vs Pitcher": group["pa_key"].nunique() if "pa_key" in group else 0,
                "Pitches": len(group),
                "Last Matchup": group["game_date"].max() if not group.empty and "game_date" in group else pd.NA,
                "Pitch Types Seen": ", ".join(group["pitch_type"].dropna().astype(str).unique()[:8]) if not group.empty else "",
                "PA Results": ", ".join(events[:8]),
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(["PA vs Pitcher", "Pitches", "Current Player"], ascending=[False, False, True])
