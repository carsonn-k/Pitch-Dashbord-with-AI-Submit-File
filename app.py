from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from pitch_staff_dashboard.ai import (
    AI_MAX_INPUT_CHARS,
    AI_MAX_OUTPUT_TOKENS,
    AI_SCOUTING_MODEL_LABEL,
    compact_frame,
    generate_scouting_brief,
)
from pitch_staff_dashboard.links import add_savant_links, add_savant_video_links
from pitch_staff_dashboard.metrics import (
    attack_plan_for_batter,
    batter_pitch_inventory,
    batter_zone_leaders,
    batter_zone_outcome_summary,
    contact_quality_by_pitch,
    pitch_result_summary,
    pitch_usage_by_count,
    pitch_usage_by_handedness,
    putaway_pitch_effectiveness,
    summarize_result_rates,
    with_pitch_flags,
)
from pitch_staff_dashboard.pitch_data import (
    ALL_PITCHES_LAST5_PARQUET_PATH,
    MLB_TEAMS,
    fetch_current_roster,
    filter_to_roster_matchups,
    hitters_from_roster,
    load_batter_rows,
    load_pitcher_rows,
    pitch_refresh_status,
    pitch_data_size_bytes,
    refresh_all_pitches_parquet,
    resolve_pitch_data_path,
    roster_matchup_table,
)
from pitch_staff_dashboard.transforms import (
    pitch_detail_table,
    plate_appearance_summary,
    prepare_pitch_sequences,
    sequence_outcome_after_pitch,
    validate_pitch_order,
)
from pitch_staff_dashboard.viz import result_bar, usage_bar, zone_heatmap


st.set_page_config(page_title="Pitcher Matchup Scout", layout="wide")

DATASET_SOURCE_LABEL = "All-pitches last-5-years Parquet"

AI_PITCH_LOG_COLUMNS = [
    "game_date",
    "at_bat_number",
    "pitch_number",
    "stand",
    "p_throws",
    "count",
    "balls",
    "strikes",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "description",
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "pa_result",
    "is_swing",
    "is_whiff",
    "is_take",
    "is_chase",
    "is_in_play",
    "is_foul",
    "is_hit",
    "is_hard_hit",
]

AI_PITCH_LOG_NUMERIC_COLUMNS = {
    "release_speed": 1,
    "release_spin_rate": 0,
    "pfx_x": 2,
    "pfx_z": 2,
    "plate_x": 2,
    "plate_z": 2,
    "launch_speed": 1,
    "launch_angle": 1,
    "estimated_woba_using_speedangle": 3,
}

AI_PITCHER_LOG_ROWS = 120
AI_BATTER_LOG_ROWS = 160
AI_DIRECT_LOG_ROWS = 80
AI_PITCHER_LOG_CHARS = 5_500
AI_BATTER_LOG_CHARS = 7_500
AI_DIRECT_LOG_CHARS = 4_000

TEAM_META: dict[str, dict[str, str]] = {
    "Arizona Diamondbacks": {"abbr": "ARI", "division": "NL West", "primary": "#A71930", "accent": "#E3D4AD"},
    "Athletics": {"abbr": "ATH", "division": "AL West", "primary": "#003831", "accent": "#EFB21E"},
    "Atlanta Braves": {"abbr": "ATL", "division": "NL East", "primary": "#13274F", "accent": "#CE1141"},
    "Baltimore Orioles": {"abbr": "BAL", "division": "AL East", "primary": "#DF4601", "accent": "#000000"},
    "Boston Red Sox": {"abbr": "BOS", "division": "AL East", "primary": "#BD3039", "accent": "#0C2340"},
    "Chicago Cubs": {"abbr": "CHC", "division": "NL Central", "primary": "#0E3386", "accent": "#CC3433"},
    "Chicago White Sox": {"abbr": "CWS", "division": "AL Central", "primary": "#27251F", "accent": "#C4CED4"},
    "Cincinnati Reds": {"abbr": "CIN", "division": "NL Central", "primary": "#C6011F", "accent": "#000000"},
    "Cleveland Guardians": {"abbr": "CLE", "division": "AL Central", "primary": "#00385D", "accent": "#E50022"},
    "Colorado Rockies": {"abbr": "COL", "division": "NL West", "primary": "#33006F", "accent": "#C4CED4"},
    "Detroit Tigers": {"abbr": "DET", "division": "AL Central", "primary": "#0C2340", "accent": "#FA4616"},
    "Houston Astros": {"abbr": "HOU", "division": "AL West", "primary": "#002D62", "accent": "#EB6E1F"},
    "Kansas City Royals": {"abbr": "KC", "division": "AL Central", "primary": "#004687", "accent": "#BD9B60"},
    "Los Angeles Angels": {"abbr": "LAA", "division": "AL West", "primary": "#BA0021", "accent": "#003263"},
    "Los Angeles Dodgers": {"abbr": "LAD", "division": "NL West", "primary": "#005A9C", "accent": "#EF3E42"},
    "Miami Marlins": {"abbr": "MIA", "division": "NL East", "primary": "#00A3E0", "accent": "#EF3340"},
    "Milwaukee Brewers": {"abbr": "MIL", "division": "NL Central", "primary": "#12284B", "accent": "#FFC52F"},
    "Minnesota Twins": {"abbr": "MIN", "division": "AL Central", "primary": "#002B5C", "accent": "#D31145"},
    "New York Mets": {"abbr": "NYM", "division": "NL East", "primary": "#002D72", "accent": "#FF5910"},
    "New York Yankees": {"abbr": "NYY", "division": "AL East", "primary": "#0C2340", "accent": "#C4CED4"},
    "Philadelphia Phillies": {"abbr": "PHI", "division": "NL East", "primary": "#E81828", "accent": "#002D72"},
    "Pittsburgh Pirates": {"abbr": "PIT", "division": "NL Central", "primary": "#27251F", "accent": "#FDB827"},
    "San Diego Padres": {"abbr": "SD", "division": "NL West", "primary": "#2F241D", "accent": "#FFC425"},
    "San Francisco Giants": {"abbr": "SF", "division": "NL West", "primary": "#FD5A1E", "accent": "#27251F"},
    "Seattle Mariners": {"abbr": "SEA", "division": "AL West", "primary": "#0C2C56", "accent": "#005C5C"},
    "St. Louis Cardinals": {"abbr": "STL", "division": "NL Central", "primary": "#C41E3A", "accent": "#0C2340"},
    "Tampa Bay Rays": {"abbr": "TB", "division": "AL East", "primary": "#092C5C", "accent": "#8FBCE6"},
    "Texas Rangers": {"abbr": "TEX", "division": "AL West", "primary": "#003278", "accent": "#C0111F"},
    "Toronto Blue Jays": {"abbr": "TOR", "division": "AL East", "primary": "#134A8E", "accent": "#E8291C"},
    "Washington Nationals": {"abbr": "WSH", "division": "NL East", "primary": "#AB0003", "accent": "#14225A"},
}

st.markdown(
    """
    <style>
    .pitch-data-card,
    .team-picker-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.8rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .pitch-data-card {
        border-left: 4px solid #246BFD;
    }
    .pitch-data-kicker,
    .team-picker-kicker {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .pitch-data-name,
    .team-picker-name {
        color: #111827;
        font-size: 0.98rem;
        font-weight: 750;
        line-height: 1.2;
        margin-top: 0.18rem;
    }
    .pitch-data-meta,
    .team-picker-meta {
        color: #475569;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }
    .team-picker-card {
        position: relative;
        overflow: hidden;
        border-left: 5px solid var(--team-primary);
    }
    .team-picker-card:before {
        background: linear-gradient(135deg, var(--team-primary), var(--team-accent));
        content: "";
        height: 100%;
        opacity: 0.08;
        position: absolute;
        right: 0;
        top: 0;
        width: 45%;
    }
    .team-picker-row {
        align-items: center;
        display: flex;
        gap: 0.72rem;
        position: relative;
    }
    .team-picker-logo {
        align-items: center;
        background: #F8FAFC;
        border: 1px solid rgba(15, 23, 42, 0.1);
        border-radius: 8px;
        display: flex;
        height: 48px;
        justify-content: center;
        width: 48px;
    }
    .team-picker-logo img {
        height: 34px;
        max-width: 38px;
    }
    .team-picker-abbr {
        background: var(--team-primary);
        border-radius: 999px;
        color: #ffffff;
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 0.45rem;
        padding: 0.16rem 0.48rem;
    }
    .team-grid-tile {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 8px;
        border-top: 4px solid var(--team-primary);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.45rem;
        min-height: 132px;
        padding: 0.62rem 0.48rem;
        text-align: center;
    }
    .team-grid-tile.is-selected {
        border-color: var(--team-primary);
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--team-primary) 24%, transparent);
    }
    .team-grid-logo {
        align-items: center;
        background: #F8FAFC;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 8px;
        display: flex;
        height: 54px;
        justify-content: center;
        margin: 0 auto 0.46rem;
        width: 54px;
    }
    .team-grid-logo img {
        max-height: 44px;
        max-width: 48px;
    }
    .team-grid-abbr {
        color: var(--team-primary);
        font-size: 0.72rem;
        font-weight: 850;
        line-height: 1;
        margin-bottom: 0.24rem;
    }
    .team-grid-name {
        color: #111827;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.15;
        min-height: 2.65rem;
        overflow-wrap: anywhere;
    }
    div[data-testid="stButton"] button {
        white-space: normal;
    }
    .pitcher-grid-tile,
    .selected-pitcher-card {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.45rem;
        padding: 0.72rem;
    }
    .pitcher-grid-tile {
        border-top: 4px solid #246BFD;
        min-height: 128px;
    }
    .pitcher-grid-tile.is-selected,
    .selected-pitcher-card {
        border-color: #246BFD;
        box-shadow: 0 0 0 2px rgba(36, 107, 253, 0.16);
    }
    .pitcher-name {
        color: #111827;
        font-size: 0.92rem;
        font-weight: 800;
        line-height: 1.15;
        min-height: 2.15rem;
    }
    .pitcher-meta {
        color: #475569;
        font-size: 0.76rem;
        font-weight: 650;
        margin-top: 0.35rem;
    }
    .pitcher-badge {
        background: #EFF6FF;
        border-radius: 999px;
        color: #1D4ED8;
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 0.5rem;
        padding: 0.16rem 0.5rem;
    }
    .matchup-hero {
        background:
            linear-gradient(135deg, rgba(18, 24, 38, 0.98) 0%, rgba(24, 29, 42, 0.98) 48%, rgba(19, 44, 42, 0.96) 100%),
            #121826;
        border: 1px solid rgba(226, 232, 240, 0.22);
        border-radius: 8px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        color: #F8FAFC;
        margin-bottom: 1rem;
        overflow: hidden;
        padding: 1.05rem;
        position: relative;
    }
    .matchup-hero:before {
        background: linear-gradient(90deg, var(--pitching-primary), #14B8A6, var(--batting-primary));
        content: "";
        height: 6px;
        left: 0;
        position: absolute;
        right: 0;
        top: 0;
    }
    .matchup-head {
        align-items: flex-start;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 0.95rem;
    }
    .matchup-eyebrow {
        color: #67E8F9;
        font-size: 0.74rem;
        font-weight: 850;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .matchup-title {
        color: #F8FAFC;
        font-size: clamp(2rem, 4.2vw, 3.3rem);
        font-weight: 900;
        letter-spacing: 0;
        line-height: 0.98;
        margin-top: 0.25rem;
    }
    .matchup-subtitle {
        color: #CBD5E1;
        font-size: 0.94rem;
        font-weight: 650;
        line-height: 1.35;
        margin-top: 0.55rem;
        max-width: 58rem;
    }
    .matchup-status-stack {
        align-items: flex-end;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: flex-end;
        max-width: 23rem;
    }
    .matchup-pill {
        background: rgba(248, 250, 252, 0.96);
        border: 1px solid rgba(226, 232, 240, 0.72);
        border-radius: 999px;
        color: #172033;
        font-size: 0.76rem;
        font-weight: 800;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
        padding: 0.24rem 0.62rem;
        white-space: nowrap;
    }
    .matchup-stage {
        align-items: stretch;
        display: grid;
        gap: 0.72rem;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        margin-bottom: 0.9rem;
    }
    .matchup-side {
        align-items: center;
        background: linear-gradient(180deg, #FFFFFF 0%, #F3F7FA 100%);
        border: 1px solid rgba(226, 232, 240, 0.78);
        border-radius: 8px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
        display: flex;
        gap: 0.75rem;
        min-width: 0;
        padding: 0.74rem;
    }
    .matchup-side.is-pitching {
        border-left: 5px solid var(--pitching-primary);
    }
    .matchup-side.is-batting {
        border-left: 5px solid var(--batting-primary);
    }
    .matchup-logo {
        align-items: center;
        background: #FFFFFF;
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 8px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.7);
        display: flex;
        flex: 0 0 58px;
        height: 58px;
        justify-content: center;
        width: 58px;
    }
    .matchup-logo img {
        max-height: 46px;
        max-width: 50px;
    }
    .matchup-role {
        color: #54657A;
        font-size: 0.72rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .matchup-name {
        color: #101827;
        font-size: 1.05rem;
        font-weight: 900;
        line-height: 1.12;
        margin-top: 0.18rem;
        overflow-wrap: anywhere;
    }
    .matchup-detail {
        color: #43536A;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.32rem;
    }
    .matchup-vs {
        align-items: center;
        color: #E2E8F0;
        display: flex;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        justify-content: center;
        min-width: 2.4rem;
    }
    .scout-metric-grid {
        display: grid;
        gap: 0.62rem;
        grid-template-columns: repeat(5, minmax(0, 1fr));
    }
    .scout-metric-tile {
        background: #FFFFFF;
        border: 1px solid rgba(226, 232, 240, 0.78);
        border-top: 4px solid #14B8A6;
        border-radius: 8px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
        min-width: 0;
        padding: 0.72rem;
    }
    .scout-metric-tile:nth-child(1) {
        border-top-color: #38BDF8;
    }
    .scout-metric-tile:nth-child(2) {
        border-top-color: #F59E0B;
    }
    .scout-metric-tile:nth-child(3) {
        border-top-color: #10B981;
    }
    .scout-metric-tile:nth-child(4) {
        border-top-color: #A78BFA;
    }
    .scout-metric-tile:nth-child(5) {
        border-top-color: #F43F5E;
    }
    .scout-metric-label {
        color: #54657A;
        font-size: 0.7rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        line-height: 1.15;
        text-transform: uppercase;
    }
    .scout-metric-value {
        color: #0B1020;
        font-size: 1.76rem;
        font-weight: 900;
        line-height: 1.05;
        margin-top: 0.34rem;
        overflow-wrap: anywhere;
    }
    .scout-metric-value.is-date {
        font-size: 1.02rem;
        line-height: 1.22;
    }
    .scout-metric-help {
        color: #54657A;
        font-size: 0.76rem;
        font-weight: 650;
        line-height: 1.25;
        margin-top: 0.34rem;
    }
    @media (max-width: 900px) {
        .matchup-head,
        .matchup-status-stack {
            align-items: flex-start;
            justify-content: flex-start;
        }
        .matchup-head {
            display: block;
        }
        .matchup-status-stack {
            margin-top: 0.78rem;
        }
        .matchup-stage,
        .scout-metric-grid {
            grid-template-columns: 1fr;
        }
        .matchup-vs {
            min-height: 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=True)
def cached_pitcher_rows(dataset_path: str, file_version: float, pitcher_id: int) -> tuple[pd.DataFrame, int]:
    return load_pitcher_rows(Path(dataset_path), pitcher_id)


@st.cache_data(show_spinner=True)
def cached_batter_rows(dataset_path: str, file_version: float, batter_id: int) -> tuple[pd.DataFrame, int]:
    return load_batter_rows(Path(dataset_path), batter_id)


@st.cache_data(show_spinner=False)
def cached_prepare(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_pitch_sequences(df)
    prepared = with_pitch_flags(prepared)
    return add_savant_links(prepared)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_roster(team_id: int, roster_type: str) -> pd.DataFrame:
    return fetch_current_roster(team_id, roster_type=roster_type)


@st.cache_data(show_spinner=False)
def cached_pitch_refresh_status(dataset_path: str, file_version: float, rolling_years: int) -> dict[str, object]:
    return pitch_refresh_status(Path(dataset_path), rolling_years=rolling_years)


def dataset_version(dataset_path: Path) -> float:
    resolved = resolve_pitch_data_path(dataset_path)
    return resolved.stat().st_mtime if resolved.exists() else 0.0


def clear_dataset_caches() -> None:
    cached_pitcher_rows.clear()
    cached_batter_rows.clear()
    cached_prepare.clear()
    cached_pitch_refresh_status.clear()
    cached_ai_brief.clear()


def format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = {"Swing %", "Whiff %", "Chase %", "Hard-Hit %", "Usage %", "Share"}
    for col in out.columns:
        if col.endswith("%") or col in percent_cols:
            out[col] = out[col].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
    return out


def pitch_type_legend(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty or not {"pitch_type", "pitch_name"}.issubset(df.columns):
        return []
    legend = (
        df[["pitch_type", "pitch_name"]]
        .dropna()
        .drop_duplicates("pitch_type")
        .sort_values("pitch_type")
    )
    return [
        {"pitch_code": str(row.pitch_type), "pitch_name": str(row.pitch_name)}
        for row in legend.itertuples(index=False)
    ]


def add_pitch_names(summary: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or source_df.empty or not {"pitch_type", "pitch_name"}.issubset(source_df.columns):
        return summary
    name_map = (
        source_df[["pitch_type", "pitch_name"]]
        .dropna()
        .drop_duplicates("pitch_type")
        .set_index("pitch_type")["pitch_name"]
        .to_dict()
    )
    out = summary.copy()
    if "pitch_type" in out.columns:
        out["pitch_name"] = out["pitch_type"].map(name_map)
    if "Pitch Type" in out.columns:
        out["Pitch Name"] = out["Pitch Type"].map(name_map)
    return out


def ai_metric_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.rename(
        columns={
            "pitch_type": "pitch_code",
            "Pitch Type": "Pitch Code",
            "Avg EV": "Avg Exit Velo MPH",
            "Max EV": "Max Exit Velo MPH",
            "Avg LA": "Avg Launch Angle Degrees",
            "AvgEV": "Avg Exit Velo MPH",
            "AvgXwOBA": "Avg xwOBA",
        }
    )


def compact_all_frame(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    return compact_frame(df, columns=columns, max_rows=len(df))


def select_ai_pitch_log_rows(
    df: pd.DataFrame,
    *,
    max_rows: int,
    preferred_pitch_types: set[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    sort_cols = [
        col
        for col in ["game_date", "game_pk", "at_bat_number", "pitch_number", "pa_pitch_index"]
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort", na_position="last")

    if preferred_pitch_types and "pitch_type" in out.columns:
        preferred = out[out["pitch_type"].astype(str).isin(preferred_pitch_types)]
        if not preferred.empty:
            out = preferred

    return out.tail(max_rows).copy()


def ai_pitch_log_csv(df: pd.DataFrame, *, max_chars: int) -> str:
    if df.empty:
        return ""

    out = df.copy()
    keep_cols = [col for col in AI_PITCH_LOG_COLUMNS if col in out.columns]
    out = out[keep_cols].copy()

    for col, decimals in AI_PITCH_LOG_NUMERIC_COLUMNS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(decimals)

    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")

    csv_lines = out.to_csv(index=False).splitlines()
    if not csv_lines:
        return ""

    selected_lines = [csv_lines[0]]
    used_chars = len(csv_lines[0]) + 1
    for line in csv_lines[1:]:
        next_chars = len(line) + 1
        if used_chars + next_chars > max_chars:
            break
        selected_lines.append(line)
        used_chars += next_chars

    if len(selected_lines) < len(csv_lines):
        selected_lines.append(f"# truncated_to_fit_character_budget,included_rows={len(selected_lines) - 1},available_rows={len(csv_lines) - 1}")

    return "\n".join(selected_lines)


def build_ai_brief_prompt(
    *,
    pitcher_name: str,
    team_name: str,
    dataset_source: str,
    data_start: object,
    data_end: object,
    pitcher_pitch_count: int,
    roster_count: int,
    matchup_df: pd.DataFrame,
    matchup_table: pd.DataFrame,
) -> str:
    matchup_sample = {
        "players_with_history": int(matchup_df["batter"].nunique()) if not matchup_df.empty else 0,
        "historical_pa": int(matchup_df["pa_key"].nunique()) if not matchup_df.empty and "pa_key" in matchup_df else 0,
        "matchup_pitches": int(len(matchup_df)),
    }
    rates = summarize_result_rates(matchup_df)
    usage = add_pitch_names(pitch_usage_by_count(matchup_df), matchup_df).sort_values(["count", "Pitches"], ascending=[True, False])
    results = pitch_result_summary(matchup_df)
    contact = add_pitch_names(contact_quality_by_pitch(matchup_df), matchup_df)
    roster_rows = matchup_table.sort_values(["PA vs Pitcher", "Pitches", "Current Player"], ascending=[False, False, True])

    return f"""
You are an MLB pitching strategist helping a coaching staff. Use only the provided matchup data.

Write a practical scouting brief with:
- Overall matchup read
- Tactical attack ideas
- Top risk hitters or say sample is thin
- Count/sequencing note if supported
- Confidence/caveat and what evidence is strongest

Do not invent facts. Do not mention data that is not provided. Be direct and baseball-specific.
Pitch code notes: use pitch_name when available. KC means Knuckle Curve. Avg Exit Velo is mph and cannot be negative. Avg Launch Angle is degrees and can be negative for ground-ball contact.

Context:
pitcher={pitcher_name}
opponent_team={team_name}
dataset={dataset_source}
pitcher_dataset_range={data_start} to {data_end}
pitcher_pitches_in_dataset={pitcher_pitch_count}
current_players_checked={roster_count}
matchup_sample={matchup_sample}
pitch_type_legend={pitch_type_legend(matchup_df)}

Roster matchup rows:
{compact_all_frame(roster_rows, columns=["Current Player", "Position", "Bats", "PA vs Pitcher", "Pitches", "Last Matchup", "Pitch Types Seen", "PA Results"])}

Result rates:
{compact_all_frame(rates)}

Pitch usage by count:
{compact_all_frame(ai_metric_labels(usage), columns=["count", "pitch_code", "pitch_name", "Pitches", "Usage %"])}

Result summary:
{compact_all_frame(results, columns=["description", "Pitches", "Share"])}

Contact quality by pitch:
{compact_all_frame(ai_metric_labels(contact))}
""".strip()


def build_ai_pitcher_batter_attack_prompt(
    *,
    pitcher_name: str,
    batter_name: str,
    team_name: str,
    dataset_source: str,
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    direct_matchup_df: pd.DataFrame,
) -> str:
    pitcher_usage_count = add_pitch_names(pitch_usage_by_count(pitcher_df), pitcher_df).sort_values(["count", "Pitches"], ascending=[True, False])
    pitcher_usage_hand = add_pitch_names(pitch_usage_by_handedness(pitcher_df), pitcher_df).sort_values(["stand", "Pitches"], ascending=[True, False])
    pitcher_contact = add_pitch_names(contact_quality_by_pitch(pitcher_df), pitcher_df)
    pitcher_putaway = add_pitch_names(putaway_pitch_effectiveness(pitcher_df), pitcher_df)

    batter_seen_usage = add_pitch_names(pitch_usage_by_count(batter_df), batter_df).sort_values(["count", "Pitches"], ascending=[True, False])
    batter_contact = add_pitch_names(contact_quality_by_pitch(batter_df), batter_df)
    batter_zone = batter_zone_outcome_summary(batter_df)
    batter_results = pitch_result_summary(batter_df)

    direct_summary = summarize_result_rates(direct_matchup_df)
    direct_usage = add_pitch_names(pitch_usage_by_count(direct_matchup_df), direct_matchup_df).sort_values(["count", "Pitches"], ascending=[True, False])
    direct_contact = add_pitch_names(contact_quality_by_pitch(direct_matchup_df), direct_matchup_df)

    pitcher_pitch_types = set(pitcher_df.get("pitch_type", pd.Series(dtype=str)).dropna().astype(str))
    pitcher_log_df = select_ai_pitch_log_rows(pitcher_df, max_rows=AI_PITCHER_LOG_ROWS)
    batter_log_df = select_ai_pitch_log_rows(
        batter_df,
        max_rows=AI_BATTER_LOG_ROWS,
        preferred_pitch_types=pitcher_pitch_types,
    )
    direct_log_df = select_ai_pitch_log_rows(direct_matchup_df, max_rows=AI_DIRECT_LOG_ROWS)

    pitcher_pitch_log = ai_pitch_log_csv(pitcher_log_df, max_chars=AI_PITCHER_LOG_CHARS)
    batter_pitch_log = ai_pitch_log_csv(batter_log_df, max_chars=AI_BATTER_LOG_CHARS)
    direct_pitch_log = ai_pitch_log_csv(direct_log_df, max_chars=AI_DIRECT_LOG_CHARS)

    context = {
        "pitcher_total_pitches": int(len(pitcher_df)),
        "pitcher_pa": int(pitcher_df["pa_key"].nunique()) if "pa_key" in pitcher_df else None,
        "batter_total_pitches_seen": int(len(batter_df)),
        "batter_pa_seen": int(batter_df["pa_key"].nunique()) if "pa_key" in batter_df else None,
        "direct_matchup_pitches": int(len(direct_matchup_df)),
        "direct_matchup_pa": int(direct_matchup_df["pa_key"].nunique()) if not direct_matchup_df.empty and "pa_key" in direct_matchup_df else 0,
        "pitcher_pitch_level_rows_included": int(len(pitcher_log_df)),
        "batter_pitch_level_rows_included": int(len(batter_log_df)),
        "direct_matchup_pitch_level_rows_included": int(len(direct_log_df)),
        "pitch_level_log_strategy": (
            "All aggregate summaries use every available row. CSV logs are smaller fast-mode recent samples; "
            "batter CSV prioritizes pitch types the selected pitcher throws."
        ),
    }

    return f"""
You are an MLB pitching strategist preparing a game-plan memo for a coaching staff.

Use only the data below. You have both:
- aggregate summaries from all available rows
- all-row aggregate summaries for the selected pitcher, selected batter, and direct matchup
- budgeted pitch-level CSV logs for the selected pitcher
- budgeted pitch-level CSV logs for the selected batter, prioritizing pitch types the selected pitcher throws
- budgeted direct matchup pitch-level CSV logs, if any

Goal:
Find the selected pitcher's strengths and the selected batter's weaknesses, then build a practical attack plan at their intersection.

Write a concise, field-usable scouting brief with these sections:
1. Primary Read
2. Pitcher Strengths To Lean On
3. Batter Weaknesses To Attack
4. Count Plan
5. Locations / Shapes To Use And Avoid
6. Put-Away / Chase Plan
7. Risk Notes
8. Confidence And Sample Caveat

Be specific but brief. Tie every recommendation to the strongest available evidence: pitch type, count, handedness, zone/location, whiff/chase, damage, contact quality, or result evidence.
Separate broader profile evidence from direct matchup evidence. Do not over-weight tiny direct matchup samples.
Do not invent pitch traits, pitch shapes, command ability, injuries, scouting grades, or outcomes not supported by the data.
Pitch code notes: use pitch_name when available. KC means Knuckle Curve. Avg Exit Velo is mph and cannot be negative. Avg Launch Angle is degrees and can be negative for ground-ball contact.
If direct_matchup_pa is small, say the direct matchup sample is small, but still use pitcher_total_pitches and batter_total_pitches_seen as broader profile evidence. Do not describe the whole plan as small-sample unless those broader samples are also small.
Prioritize the broader pitcher profile and batter pitch-seen profile when direct matchup history is thin.
Treat aggregate summaries as the source of truth for full-sample rates. Use pitch-level CSV logs for granular examples, recency, sequencing, counts, and location detail.

Context:
pitcher={pitcher_name}
batter={batter_name}
batter_team={team_name}
dataset={dataset_source}
sample_counts={context}
pitcher_pitch_type_legend={pitch_type_legend(pitcher_df)}
batter_seen_pitch_type_legend={pitch_type_legend(batter_df)}

Pitcher overall result rates:
{compact_all_frame(summarize_result_rates(pitcher_df))}

Pitcher usage by count:
{compact_all_frame(ai_metric_labels(pitcher_usage_count), columns=["count", "pitch_code", "pitch_name", "Pitches", "Usage %"])}

Pitcher usage by batter handedness:
{compact_all_frame(ai_metric_labels(pitcher_usage_hand), columns=["stand", "pitch_code", "pitch_name", "Pitches", "Usage %"])}

Pitcher contact quality by pitch:
{compact_all_frame(ai_metric_labels(pitcher_contact))}

Pitcher two-strike/put-away pitches:
{compact_all_frame(ai_metric_labels(pitcher_putaway))}

Batter all-pitch result rates:
{compact_all_frame(summarize_result_rates(batter_df))}

Batter pitches seen by count:
{compact_all_frame(ai_metric_labels(batter_seen_usage), columns=["count", "pitch_code", "pitch_name", "Pitches", "Usage %"])}

Batter contact quality by pitch type seen:
{compact_all_frame(ai_metric_labels(batter_contact))}

Batter zone outcomes:
{compact_all_frame(ai_metric_labels(batter_zone), columns=["zone", "Pitches", "Swing %", "Whiff/Swing %", "Hit/BIP %", "Hard-Hit/BIP %", "Avg Exit Velo MPH", "Avg xwOBA"])}

Batter result summary:
{compact_all_frame(batter_results, columns=["description", "Pitches", "Share"])}

Direct matchup result rates:
{compact_all_frame(direct_summary)}

Direct matchup pitch usage:
{compact_all_frame(ai_metric_labels(direct_usage), columns=["count", "pitch_code", "pitch_name", "Pitches", "Usage %"])}

Direct matchup contact quality:
{compact_all_frame(ai_metric_labels(direct_contact))}

Budgeted selected-pitcher pitch-level CSV:
```csv
{pitcher_pitch_log}
```

Budgeted selected-batter pitch-level CSV:
```csv
{batter_pitch_log}
```

Budgeted direct matchup pitch-level CSV:
```csv
{direct_pitch_log}
```
""".strip()


@st.cache_data(show_spinner=False)
def cached_ai_brief(
    prompt: str,
    file_version: float,
    model_label: str,
    max_input_chars: int,
    max_output_tokens: int,
) -> str:
    return generate_scouting_brief(prompt)


def batter_select_options(matchups: pd.DataFrame, roster: pd.DataFrame, pitcher_name: str) -> dict[str, int | None]:
    options: dict[str, int | None] = {f"All current players with {pitcher_name} history": None}
    if matchups.empty:
        return options

    counts = (
        matchups.groupby(["batter", "Current Player"], dropna=False)
        .agg(PA=("pa_key", "nunique"), Pitches=("pitch_number", "count"))
        .reset_index()
        .sort_values(["PA", "Pitches", "Current Player"], ascending=[False, False, True])
    )
    for _, row in counts.iterrows():
        options[f"{row['Current Player']} ({int(row['PA'])} PA, {int(row['Pitches'])} pitches)"] = int(row["batter"])
    return options


def roster_batter_profile_options(roster: pd.DataFrame, matchups: pd.DataFrame, pitcher_name: str) -> dict[str, int]:
    options: dict[str, int] = {}
    if roster.empty:
        return options

    matchup_counts: dict[int, tuple[int, int]] = {}
    if not matchups.empty:
        counts = (
            matchups.groupby(["batter"], dropna=False)
            .agg(PA=("pa_key", "nunique"), Pitches=("pitch_number", "count"))
            .reset_index()
        )
        for row in counts.itertuples(index=False):
            if pd.notna(row.batter):
                matchup_counts[int(row.batter)] = (int(row.PA), int(row.Pitches))

    sorted_roster = roster.sort_values(["Current Player", "batter"])
    for _, player in sorted_roster.iterrows():
        batter_id = int(player["batter"])
        pa, pitches = matchup_counts.get(batter_id, (0, 0))
        player_name = player.get("Current Player") or f"Batter {batter_id}"
        label = f"{player_name} ({pa} PA vs {pitcher_name}, {pitches} pitches)"
        options[label] = batter_id
    return options


def team_logo_url(team_name: str) -> str:
    return f"https://www.mlbstatic.com/team-logos/{MLB_TEAMS[team_name]}.svg"


def selected_team_name(state_key: str, default_team: str = "Arizona Diamondbacks") -> str:
    team_name = st.session_state.get(state_key, default_team)
    if team_name not in MLB_TEAMS:
        team_name = default_team
    st.session_state[state_key] = team_name
    return team_name


def render_team_grid_tile(team_name: str, selected: bool = False) -> None:
    meta = TEAM_META.get(team_name, {})
    abbr = meta.get("abbr", team_name[:3].upper())
    primary = meta.get("primary", "#1F2937")
    selected_class = " is-selected" if selected else ""
    st.markdown(
        f"""
        <div class="team-grid-tile{selected_class}" style="--team-primary: {primary};">
            <div class="team-grid-logo">
                <img src="{team_logo_url(team_name)}" alt="{team_name} logo">
            </div>
            <div class="team-grid-abbr">{abbr}</div>
            <div class="team-grid-name">{team_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Select Team", width="large")
def team_picker_dialog(state_key: str, picker_context: str) -> None:
    st.caption(picker_context)
    current_team = selected_team_name(state_key)
    teams = list(MLB_TEAMS.keys())
    for row_start in range(0, len(teams), 5):
        cols = st.columns(5)
        for col, team_name in zip(cols, teams[row_start : row_start + 5]):
            selected = team_name == current_team
            with col:
                render_team_grid_tile(team_name, selected=selected)
                if st.button(
                    "Selected" if selected else "Choose",
                    key=f"choose_team_{state_key}_{MLB_TEAMS[team_name]}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                    disabled=selected,
                ):
                    st.session_state[state_key] = team_name
                    st.rerun()


def team_board_selector(*, state_key: str, button_label: str, picker_context: str) -> str:
    team_name = selected_team_name(state_key)
    if st.button(
        button_label,
        key=f"open_team_picker_{state_key}",
        type="primary",
        icon=":material/grid_view:",
        use_container_width=True,
    ):
        team_picker_dialog(state_key, picker_context)
    return team_name


def pitchers_from_team_roster(roster: pd.DataFrame) -> pd.DataFrame:
    columns = ["pitcher", "Pitcher", "Throws", "Position", "Pitches", "Has Data"]
    if roster.empty:
        return pd.DataFrame(columns=columns)

    position = roster.get("Position", pd.Series("", index=roster.index)).fillna("").astype(str)
    position_type = roster.get("Position Type", pd.Series("", index=roster.index)).fillna("").astype(str)
    pitcher_roster = roster[position.eq("P") | position_type.eq("Pitcher")].copy()
    if pitcher_roster.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for _, player in pitcher_roster.sort_values(["Current Player", "batter"]).iterrows():
        player_id = pd.to_numeric(player.get("batter"), errors="coerce")
        if pd.isna(player_id):
            continue
        pitcher_id = int(player_id)
        rows.append(
            {
                "pitcher": pitcher_id,
                "Pitcher": player.get("Current Player") or f"Pitcher {pitcher_id}",
                "Throws": player.get("Throws", ""),
                "Position": player.get("Position", ""),
                "Pitches": pd.NA,
                "Has Data": True,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["Pitcher", "pitcher"], ascending=[True, True]).reset_index(drop=True)


def selected_pitcher_record(team_pitchers: pd.DataFrame) -> dict[str, object] | None:
    if team_pitchers.empty:
        st.session_state.pop("selected_pitcher_id", None)
        return None

    available_ids = set(pd.to_numeric(team_pitchers["pitcher"], errors="coerce").dropna().astype(int))
    preferred = team_pitchers[team_pitchers["Has Data"].astype(bool)]
    default_row = preferred.iloc[0] if not preferred.empty else team_pitchers.iloc[0]
    current_id = st.session_state.get("selected_pitcher_id")
    if current_id is None or int(current_id) not in available_ids:
        st.session_state["selected_pitcher_id"] = int(default_row["pitcher"])

    selected_id = int(st.session_state["selected_pitcher_id"])
    selected_rows = team_pitchers[pd.to_numeric(team_pitchers["pitcher"], errors="coerce").eq(selected_id)]
    if selected_rows.empty:
        selected_rows = pd.DataFrame([default_row])
        st.session_state["selected_pitcher_id"] = int(default_row["pitcher"])
    return selected_rows.iloc[0].to_dict()


def render_pitcher_tile(pitcher: dict[str, object], selected: bool = False) -> None:
    selected_class = " is-selected" if selected else ""
    name = str(pitcher.get("Pitcher", "Pitcher"))
    throws = str(pitcher.get("Throws", "") or "unknown")
    pitches = pd.to_numeric(pd.Series([pitcher.get("Pitches")]), errors="coerce").iloc[0]
    if pd.notna(pitches) and int(pitches) > 0:
        badge = f"{int(pitches):,} local pitches"
    elif pd.notna(pitches):
        badge = "No local pitch data"
    else:
        badge = "Loads after selection"
    st.markdown(
        f"""
        <div class="pitcher-grid-tile{selected_class}">
            <div class="pitcher-name">{html_text(name)}</div>
            <div class="pitcher-meta">Throws {html_text(throws)}</div>
            <span class="pitcher-badge">{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_pitcher_card(pitcher: dict[str, object]) -> None:
    render_pitcher_tile(pitcher, selected=True)


def pitcher_option_label(pitcher: dict[str, object]) -> str:
    name = str(pitcher.get("Pitcher", "Pitcher"))
    pitcher_id = int(pitcher.get("pitcher", 0) or 0)
    details = []
    throws = str(pitcher.get("Throws", "") or "").strip()
    position = str(pitcher.get("Position", "") or "").strip()
    if throws:
        details.append(f"Throws {throws}")
    if position:
        details.append(position)
    details.append(f"MLBAM {pitcher_id}")
    return f"{name} - {' | '.join(details)}"


@st.dialog("Select Pitcher", width="large")
def pitcher_picker_dialog(pitchers: list[dict[str, object]], team_name: str) -> None:
    st.caption(f"{team_name} roster pitchers")
    if not pitchers:
        st.info("No pitchers were found on this roster.")
        return

    current_id = st.session_state.get("selected_pitcher_id")
    for row_start in range(0, len(pitchers), 3):
        cols = st.columns(3)
        for col, pitcher in zip(cols, pitchers[row_start : row_start + 3]):
            pitcher_id = int(pitcher.get("pitcher", 0))
            has_data = bool(pitcher.get("Has Data", False))
            selected = current_id is not None and int(current_id) == pitcher_id
            with col:
                render_pitcher_tile(pitcher, selected=selected)
                if st.button(
                    "Selected" if selected else "Choose" if has_data else "No data",
                    key=f"choose_pitcher_{team_name}_{pitcher_id}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                    disabled=selected or not has_data,
                ):
                    st.session_state["selected_pitcher_id"] = pitcher_id
                    st.rerun()


def pitcher_board_selector(team_pitchers: pd.DataFrame, team_name: str, roster_type: str) -> dict[str, object] | None:
    if team_pitchers.empty:
        return None
    pitchers = team_pitchers.to_dict("records")
    options = {pitcher_option_label(pitcher): pitcher for pitcher in pitchers}
    selected_label = st.selectbox(
        "Pitcher",
        list(options.keys()),
        index=None,
        placeholder="Choose a pitcher to load pitch history",
        key=f"pitcher_select_{MLB_TEAMS[team_name]}_{roster_type}",
    )
    if selected_label is None:
        st.info("Choose a pitcher to load only that player's local pitch history.")
        return None
    pitcher = options[selected_label]
    st.session_state["selected_pitcher_id"] = int(pitcher.get("pitcher", 0))
    render_selected_pitcher_card(pitcher)
    return pitcher


def render_dataset_card(dataset_path: Path, resolved_path: Path) -> None:
    if resolved_path.exists():
        file_mb = pitch_data_size_bytes(resolved_path) / 1_000_000
        modified = date.fromtimestamp(resolved_path.stat().st_mtime)
        status = f"{file_mb:.1f} MB - modified {modified}"
    else:
        status = "Build required before scouting"
    st.markdown(
        f"""
        <div class="pitch-data-card">
            <div class="pitch-data-kicker">Pitch data</div>
            <div class="pitch-data-name">Last 5 years</div>
            <div class="pitch-data-meta">{status}</div>
            <div class="pitch-data-meta"><code>{dataset_path.name}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_team_card(team_name: str) -> None:
    meta = TEAM_META.get(team_name, {})
    primary = meta.get("primary", "#1F2937")
    accent = meta.get("accent", "#CBD5E1")
    abbr = meta.get("abbr", team_name[:3].upper())
    division = meta.get("division", "MLB")
    team_id = MLB_TEAMS[team_name]
    logo_url = f"https://www.mlbstatic.com/team-logos/{team_id}.svg"
    st.markdown(
        f"""
        <div class="team-picker-card" style="--team-primary: {primary}; --team-accent: {accent};">
            <div class="team-picker-row">
                <div class="team-picker-logo">
                    <img src="{logo_url}" alt="{team_name} logo">
                </div>
                <div>
                    <div class="team-picker-kicker">{division}</div>
                    <div class="team-picker-name">{team_name}</div>
                    <span class="team-picker-abbr">{abbr}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def html_text(value: object) -> str:
    return escape(str(value or ""))


def display_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return f"{parsed.strftime('%b')} {parsed.day}, {parsed.year}"
    text = str(value or "").strip()
    return text if text else "Unknown"


def refresh_status_text(refresh_result: dict[str, object]) -> tuple[str, str]:
    status = refresh_result.get("status")
    if status == "up_to_date":
        return "Current", "Statcast freshness is up to date"
    if status == "stale":
        missing_days = int(refresh_result.get("missing_days", 0) or 0)
        day_label = "day" if missing_days == 1 else "days"
        max_date = refresh_result.get("max_date", "unknown")
        return "Stale", f"{missing_days} {day_label} behind | latest local game {max_date}"
    if status == "failed":
        return "Check failed", "Freshness check did not complete"
    if status == "not_checked":
        return "Not checked", "Use the sidebar to check Statcast freshness"
    return "Unknown", "Freshness status unavailable"


def render_matchup_overview(
    *,
    pitching_team_name: str,
    batting_team_name: str,
    selected_pitcher: dict[str, object],
    roster_label: str,
    dataset_source: str,
    refresh_result: dict[str, object],
    pitch_count: int,
    data_start: object,
    data_end: object,
    batters_checked: int,
    players_with_history: int,
    historical_pa: int,
) -> None:
    pitching_meta = TEAM_META.get(pitching_team_name, {})
    batting_meta = TEAM_META.get(batting_team_name, {})
    pitching_primary = pitching_meta.get("primary", "#1F2937")
    batting_primary = batting_meta.get("primary", "#1F2937")
    pitcher_name = str(selected_pitcher.get("Pitcher", "Selected pitcher") or "Selected pitcher")
    pitcher_throws = str(selected_pitcher.get("Throws", "") or "unknown")
    date_range = f"{display_date(data_start)} to {display_date(data_end)}"
    coverage_pct = (players_with_history / batters_checked * 100) if batters_checked else 0
    status_value, status_help = refresh_status_text(refresh_result)

    metrics = [
        ("Pitcher Sample", f"{pitch_count:,}", "Pitch-level rows for this pitcher", ""),
        ("Date Range", date_range, "Selected pitcher's local history", " is-date"),
        ("Roster Coverage", f"{players_with_history}/{batters_checked}", f"{coverage_pct:.0f}% of checked hitters have history", ""),
        ("Historical PA", f"{historical_pa:,}", "Plate appearances vs current roster bats", ""),
        ("Data Status", status_value, status_help, " is-date"),
    ]
    metric_html = "\n".join(
        f"""
        <div class="scout-metric-tile">
            <div class="scout-metric-label">{html_text(label)}</div>
            <div class="scout-metric-value{value_class}">{html_text(value)}</div>
            <div class="scout-metric-help">{html_text(help_text)}</div>
        </div>
        """
        for label, value, help_text, value_class in metrics
    )

    st.html(
        f"""
        <section class="matchup-hero" style="--pitching-primary: {pitching_primary}; --batting-primary: {batting_primary};">
            <div class="matchup-head">
                <div>
                    <div class="matchup-eyebrow">Pitcher matchup dashboard</div>
                    <div class="matchup-title">Pitcher Matchup Scout</div>
                    <div class="matchup-subtitle">
                        {html_text(pitcher_name)} matchup view against the {html_text(batting_team_name)} current roster.
                    </div>
                </div>
                <div class="matchup-status-stack">
                    <span class="matchup-pill">{html_text(dataset_source)}</span>
                    <span class="matchup-pill">{html_text(roster_label)}</span>
                    <span class="matchup-pill">{html_text(status_value)}</span>
                </div>
            </div>
            <div class="matchup-stage">
                <div class="matchup-side is-pitching">
                    <div class="matchup-logo">
                        <img src="{team_logo_url(pitching_team_name)}" alt="{html_text(pitching_team_name)} logo">
                    </div>
                    <div>
                        <div class="matchup-role">Pitching team</div>
                        <div class="matchup-name">{html_text(pitching_team_name)}</div>
                        <div class="matchup-detail">{html_text(pitcher_name)} | Throws {html_text(pitcher_throws)}</div>
                    </div>
                </div>
                <div class="matchup-vs">VS</div>
                <div class="matchup-side is-batting">
                    <div class="matchup-logo">
                        <img src="{team_logo_url(batting_team_name)}" alt="{html_text(batting_team_name)} logo">
                    </div>
                    <div>
                        <div class="matchup-role">Opponent roster</div>
                        <div class="matchup-name">{html_text(batting_team_name)}</div>
                        <div class="matchup-detail">{batters_checked:,} hitters checked | {html_text(roster_label)}</div>
                    </div>
                </div>
            </div>
            <div class="scout-metric-grid">
                {metric_html}
            </div>
        </section>
        """,
    )


def show_missing_dataset(dataset_path: Path, source_label: str) -> None:
    st.warning(f"No `{source_label}` Parquet dataset found at `{dataset_path}`.")
    st.write("Build it once, then the app will read that local Parquet file and filter it.")
    st.code("python scripts/build_all_pitches_parquet.py --overwrite", language="bash")
    st.caption("Use the terminal command above for the large all-pitches dataset so progress is visible.")
    st.stop()


dataset_source = DATASET_SOURCE_LABEL
selected_dataset_path = ALL_PITCHES_LAST5_PARQUET_PATH
resolved_dataset_path = resolve_pitch_data_path(selected_dataset_path)
rolling_years = 5

with st.sidebar:
    st.header("Data")
    render_dataset_card(selected_dataset_path, resolved_dataset_path)

if not resolved_dataset_path.exists():
    show_missing_dataset(selected_dataset_path, dataset_source)

with st.sidebar:
    st.header("Pitching Team")
    pitching_team_name = team_board_selector(
        state_key="selected_pitching_team",
        button_label="Select Pitching Team",
        picker_context="Choose the team whose pitcher is starting or available for this game.",
    )
    render_team_card(pitching_team_name)
    roster_label = st.radio("Roster scope", ["Active roster", "40-man roster"], horizontal=False)
    roster_type = "active" if roster_label == "Active roster" else "40Man"

try:
    pitching_roster = cached_roster(MLB_TEAMS[pitching_team_name], roster_type)
except Exception as exc:
    st.error(str(exc))
    st.stop()

team_pitchers = pitchers_from_team_roster(pitching_roster)

with st.sidebar:
    st.header("Pitcher")
    selected_pitcher = pitcher_board_selector(team_pitchers, pitching_team_name, roster_type)

if selected_pitcher is None:
    st.info(f"Choose a {pitching_team_name} pitcher to load pitch history and opponent matchups.")
    st.stop()

selected_pitcher_id = int(selected_pitcher["pitcher"])
selected_pitcher_name = str(selected_pitcher["Pitcher"])

try:
    raw_df, raw_dataset_rows = cached_pitcher_rows(str(resolved_dataset_path), dataset_version(resolved_dataset_path), selected_pitcher_id)
except Exception as exc:
    st.error(f"Could not load pitcher rows from selected dataset: {exc}")
    st.stop()

if raw_df.empty:
    st.error(f"The selected dataset has no rows for {selected_pitcher_name} / MLBAM {selected_pitcher_id}.")
    st.stop()

selected_pitcher["Pitches"] = len(raw_df)

refresh_signature = f"{resolved_dataset_path}:{dataset_version(resolved_dataset_path)}:{rolling_years}"
if st.session_state.get("pitch_refresh_signature") != refresh_signature:
    st.session_state["pitch_refresh_signature"] = refresh_signature
    st.session_state.pop("pitch_refresh_result", None)
refresh_result = st.session_state.get("pitch_refresh_result", {"status": "not_checked"})
with st.sidebar:
    st.header("Dataset Freshness")
    if st.button("Check Statcast Freshness", use_container_width=True):
        with st.spinner("Checking local dataset dates..."):
            try:
                refresh_result = cached_pitch_refresh_status(
                    str(resolved_dataset_path),
                    dataset_version(resolved_dataset_path),
                    rolling_years,
                )
            except Exception as exc:
                refresh_result = {"status": "failed", "error": str(exc)}
            st.session_state["pitch_refresh_result"] = refresh_result
    status = refresh_result.get("status")
    if status == "up_to_date":
        st.caption("Statcast freshness: up to date.")
    elif status == "stale":
        missing_days = int(refresh_result.get("missing_days", 0))
        max_date = refresh_result.get("max_date")
        st.warning(f"Local data is {missing_days} day(s) behind Statcast. Latest game date: {max_date}.")
        if st.button("Refresh Recent Statcast Data"):
            with st.spinner("Fetching recent pitch data..."):
                result = refresh_all_pitches_parquet(resolved_dataset_path, rolling_years=rolling_years)
            if result.get("status") == "updated":
                clear_dataset_caches()
                st.success(f"Updated {int(result.get('rows_added', 0)):,} row(s).")
                st.rerun()
            elif result.get("status") == "up_to_date":
                st.success("Dataset is already up to date.")
            elif result.get("status") == "no_new_rows":
                st.info(f"No newer Statcast rows were available for {result.get('start')} to {result.get('end')}.")
            else:
                st.warning(f"Refresh result: {result}")
    elif status == "failed":
        st.warning(f"Freshness check failed: {refresh_result.get('error')}")
    elif status == "not_checked":
        st.caption("Freshness check is manual so matchup loading stays fast.")
    else:
        st.caption("Statcast freshness: unknown.")

df = cached_prepare(raw_df)

if df.empty:
    st.error(f"No prepared pitch rows for {selected_pitcher_name} / MLBAM {selected_pitcher_id}.")
    st.stop()

order_issues = validate_pitch_order(df)
if not order_issues.empty:
    st.warning(f"{len(order_issues)} plate appearance(s) have pitch-number ordering issues.")

with st.sidebar:
    st.header("Batting Team")
    team_name = team_board_selector(
        state_key="selected_batting_team",
        button_label="Select Batting Team",
        picker_context="Choose the opponent lineup this pitcher is going to face.",
    )
    render_team_card(team_name)
    if team_name == pitching_team_name:
        st.warning("Pitching and batting teams are the same.")
    include_pitchers = st.checkbox("Include pitchers in roster table", value=False)

try:
    roster = cached_roster(MLB_TEAMS[team_name], roster_type)
except Exception as exc:
    st.error(str(exc))
    st.stop()

scouting_roster = hitters_from_roster(roster, include_pitchers=include_pitchers)
matchup_df = filter_to_roster_matchups(df, scouting_roster)
matchup_table = roster_matchup_table(matchup_df, scouting_roster)

if "game_date" in df and df["game_date"].notna().any():
    data_start = min(df["game_date"].dropna())
    data_end = max(df["game_date"].dropna())
else:
    data_start = data_end = "unknown"

players_with_history = matchup_df["batter"].nunique() if not matchup_df.empty else 0
historical_pa = matchup_df["pa_key"].nunique() if not matchup_df.empty else 0
render_matchup_overview(
    pitching_team_name=pitching_team_name,
    batting_team_name=team_name,
    selected_pitcher=selected_pitcher,
    roster_label=roster_label,
    dataset_source=dataset_source,
    refresh_result=refresh_result,
    pitch_count=len(df),
    data_start=data_start,
    data_end=data_end,
    batters_checked=len(scouting_roster),
    players_with_history=players_with_history,
    historical_pa=historical_pa,
)

tabs = st.tabs(["Roster Matchups", "At-Bats Pitch By Pitch", "Batter Profile", "AI Brief", "Dataset"])

with tabs[0]:
    st.subheader(f"{team_name} Current Roster vs {selected_pitcher_name}")
    st.caption("Rows with 0 PA are current roster players who have no pitch-level Statcast matchup history against the selected pitcher in this dataset.")
    st.dataframe(matchup_table, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("At-Bat Sequences")
    if matchup_df.empty:
        st.info(f"No current {team_name} players have matchups against {selected_pitcher_name} in the local dataset.")
    else:
        options = batter_select_options(matchup_df, scouting_roster, selected_pitcher_name)
        selected_label = st.selectbox("Batter", list(options.keys()))
        selected_batter = options[selected_label]
        selected_df = matchup_df.copy() if selected_batter is None else matchup_df[pd.to_numeric(matchup_df["batter"], errors="coerce").eq(selected_batter)]

        pa_table = plate_appearance_summary(selected_df)
        st.dataframe(pa_table, use_container_width=True, hide_index=True)

        pa_choices = {
            f"{row['Date']} {row['Batter']} G{row['Game']} PA {idx + 1}: {row['PA Result']}": row["pa_key"]
            for idx, row in pa_table.iterrows()
        }
        if pa_choices:
            selected_pa_label = st.selectbox("Plate appearance", list(pa_choices.keys()))
            selected_pa_key = pa_choices[selected_pa_label]
            pa_df = selected_df[selected_df["pa_key"].eq(selected_pa_key)].copy()
            pa_df = add_savant_video_links(pa_df, max_games=1)

            st.markdown("**Pitch Sequence**")
            detail = pitch_detail_table(pa_df)
            st.dataframe(
                detail,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "savant_url": st.column_config.LinkColumn("Savant", display_text="pitch"),
                    "estimated_woba_using_speedangle": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                    "release_speed": st.column_config.NumberColumn("Velo", format="%.1f"),
                    "release_spin_rate": st.column_config.NumberColumn("Spin", format="%.0f"),
                },
            )

            pitch_labels = [
                f"P{int(row['pitch_number'])} {row['count']} {row['pitch_type']} {row['description']}"
                for _, row in pa_df.sort_values("pitch_number").iterrows()
            ]
            selected_pitch_label = st.selectbox("What happened after pitch...", pitch_labels)
            selected_pitch_number = int(selected_pitch_label.split()[0].replace("P", ""))
            after = sequence_outcome_after_pitch(pa_df, selected_pitch_number)
            if after.empty:
                st.write("That was the final pitch of the PA.")
            else:
                st.dataframe(
                    after,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "savant_url": st.column_config.LinkColumn("Savant", display_text="pitch"),
                    },
                )

with tabs[2]:
    st.subheader("Batter Profile")
    player_options = roster_batter_profile_options(scouting_roster, matchup_df, selected_pitcher_name)
    if not player_options:
        st.info("No current roster batters are available for this team/roster selection.")
    else:
        selected_profile_label = st.selectbox("Profile batter", list(player_options.keys()))
        profile_batter = player_options[selected_profile_label]
        profile_name = scouting_roster.loc[
            pd.to_numeric(scouting_roster["batter"], errors="coerce").eq(profile_batter),
            "Current Player",
        ].iloc[0]
        profile_df = matchup_df[pd.to_numeric(matchup_df["batter"], errors="coerce").eq(profile_batter)]

        profile_tabs = st.tabs([f"Vs {selected_pitcher_name}", "All Dataset Pitches"])

        with profile_tabs[0]:
            st.markdown(f"**{profile_name} vs {selected_pitcher_name}**")
            if profile_df.empty:
                st.info(f"No pitch-level matchup history found for {profile_name} vs {selected_pitcher_name} in this dataset.")
            else:
                st.markdown("**Attack Plan Starter**")
                st.write(attack_plan_for_batter(profile_df, profile_batter))

                summary = summarize_result_rates(profile_df)
                if not summary.empty:
                    st.dataframe(format_percent_columns(summary), use_container_width=True, hide_index=True)

                left, right = st.columns(2)
                with left:
                    usage = pitch_usage_by_count(profile_df)
                    st.altair_chart(usage_bar(usage, x="count", title="Pitch Mix Seen By Count"), use_container_width=True)
                    st.dataframe(format_percent_columns(usage), use_container_width=True, hide_index=True)
                with right:
                    results = pitch_result_summary(profile_df)
                    st.altair_chart(result_bar(results), use_container_width=True)
                    st.dataframe(format_percent_columns(results), use_container_width=True, hide_index=True)

                st.markdown("**Contact Quality**")
                st.dataframe(format_percent_columns(contact_quality_by_pitch(profile_df)), use_container_width=True, hide_index=True)
                st.altair_chart(zone_heatmap(profile_df, "Avg xwOBA"), use_container_width=False)

                st.markdown("**Pitch Log vs Selected Pitcher**")
                matchup_inventory = batter_pitch_inventory(add_savant_video_links(profile_df))
                st.dataframe(
                    matchup_inventory,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "savant_url": st.column_config.LinkColumn("Savant", display_text="pitch"),
                        "release_speed": st.column_config.NumberColumn("Velo", format="%.1f"),
                        "release_spin_rate": st.column_config.NumberColumn("Spin", format="%.0f"),
                        "estimated_woba_using_speedangle": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                    },
                )

        with profile_tabs[1]:
            st.markdown(f"**Every Pitch {profile_name} Has Seen In `{dataset_source}`**")
            st.caption("This filters the selected dataset by batter ID and includes all pitchers in that file, not just the selected pitcher.")
            load_all_dataset = st.toggle(
                "Load all dataset pitches for this batter",
                value=False,
                key=f"load_all_batter_{selected_dataset_path.name}_{profile_batter}",
            )
            if not load_all_dataset:
                st.info("Turn this on to filter the selected dataset for this batter across all pitchers.")
            else:
                try:
                    all_batter_raw, batter_dataset_rows = cached_batter_rows(
                        str(resolved_dataset_path),
                        dataset_version(resolved_dataset_path),
                        profile_batter,
                    )
                except Exception as exc:
                    st.error(f"Could not load all batter rows from selected dataset: {exc}")
                    all_batter_raw = pd.DataFrame()
                    batter_dataset_rows = 0

                if all_batter_raw.empty:
                    st.info(f"No pitches for {profile_name} were found anywhere in the selected dataset.")
                else:
                    all_batter_df = cached_prepare(all_batter_raw)
                    if "batter_display" in all_batter_df:
                        all_batter_df["batter_display"] = all_batter_df["batter_display"].fillna(profile_name)

                    if "game_date" in all_batter_df and all_batter_df["game_date"].notna().any():
                        batter_start = min(all_batter_df["game_date"].dropna())
                        batter_end = max(all_batter_df["game_date"].dropna())
                    else:
                        batter_start = batter_end = "unknown"

                    batter_metrics = st.columns(5)
                    batter_metrics[0].metric("Pitches Seen", f"{len(all_batter_df):,}")
                    batter_metrics[1].metric("Plate Appearances", f"{all_batter_df['pa_key'].nunique():,}")
                    batter_metrics[2].metric("Pitchers Faced", f"{all_batter_df['pitcher'].nunique():,}")
                    batter_metrics[3].metric("Dataset Rows", f"{batter_dataset_rows:,}")
                    batter_metrics[4].metric("Date Range", f"{batter_start} to {batter_end}")

                    all_pitch_tabs = st.tabs(["Pitch Log", "Graphical Zones", "Zone Tables"])
                    with all_pitch_tabs[0]:
                        inventory = batter_pitch_inventory(add_savant_video_links(all_batter_df, max_games=8))
                        st.dataframe(
                            inventory,
                            use_container_width=True,
                            hide_index=True,
                            height=520,
                            column_config={
                                "savant_url": st.column_config.LinkColumn("Savant", display_text="pitch"),
                                "release_speed": st.column_config.NumberColumn("Velo", format="%.1f"),
                                "release_spin_rate": st.column_config.NumberColumn("Spin", format="%.0f"),
                                "estimated_woba_using_speedangle": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                            },
                        )

                    with all_pitch_tabs[1]:
                        st.caption("Heatmaps use actual plate locations. Darker/hotter cells show where the selected outcome happened most often.")
                        heatmap_tabs = st.tabs(["Hits", "Whiffs", "Fouls", "Takes", "Damage"])
                        heatmap_metrics = ["Hits", "Whiffs", "Fouls", "Takes", "Avg xwOBA"]
                        for heatmap_tab, heatmap_metric in zip(heatmap_tabs, heatmap_metrics):
                            with heatmap_tab:
                                st.altair_chart(zone_heatmap(all_batter_df, heatmap_metric), use_container_width=False)

                    with all_pitch_tabs[2]:
                        zone_summary = batter_zone_outcome_summary(all_batter_df)
                        st.dataframe(format_percent_columns(zone_summary), use_container_width=True, hide_index=True)

                        zone_tabs = st.tabs(["Hits", "Whiffs", "Fouls", "Takes", "Damage"])
                        leaders = batter_zone_leaders(all_batter_df)
                        leader_keys = ["hits", "whiffs", "fouls", "takes", "damage"]
                        for zone_tab, key in zip(zone_tabs, leader_keys):
                            with zone_tab:
                                st.dataframe(format_percent_columns(leaders[key].head(8)), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader(f"AI Scouting Brief: {selected_pitcher_name} vs {team_name}")
    ai_budget_signature = f"{AI_SCOUTING_MODEL_LABEL}:{AI_MAX_INPUT_CHARS}:{AI_MAX_OUTPUT_TOKENS}"
    st.caption(
        f"Fast mode: {AI_SCOUTING_MODEL_LABEL}. Uses all-row summaries plus smaller pitch-level samples. "
        f"Prompt cap: {AI_MAX_INPUT_CHARS:,} chars; "
        f"response cap: {AI_MAX_OUTPUT_TOKENS:,} tokens."
    )

    st.markdown("**Pitcher vs Batter Attack Plan**")
    ai_batter_options = roster_batter_profile_options(scouting_roster, matchup_df, selected_pitcher_name)
    if not ai_batter_options:
        st.info("No current roster batters are available for an AI attack plan.")
    else:
        ai_batter_label = st.selectbox("Attack-plan batter", list(ai_batter_options.keys()), key="ai_attack_batter")
        ai_batter_id = ai_batter_options[ai_batter_label]
        ai_batter_name = scouting_roster.loc[
            pd.to_numeric(scouting_roster["batter"], errors="coerce").eq(ai_batter_id),
            "Current Player",
        ].iloc[0]
        attack_key = (
            f"ai_attack:{resolved_dataset_path}:{dataset_version(resolved_dataset_path)}:"
            f"{pitching_team_name}:{selected_pitcher_id}:{ai_batter_id}:{team_name}:{roster_type}:{include_pitchers}:"
            f"{ai_budget_signature}"
        )
        st.caption("Uses every selected-pitcher and selected-batter pitch for summaries, then sends smaller recent pitch-level samples for granular evidence.")
        if st.button("Generate Pitcher-Batter Attack Plan", type="primary"):
            try:
                with st.spinner("Loading batter history and generating fast attack plan..."):
                    batter_raw, _ = cached_batter_rows(
                        str(resolved_dataset_path),
                        dataset_version(resolved_dataset_path),
                        ai_batter_id,
                    )
                    batter_all_df = cached_prepare(batter_raw)
                    direct_matchup_df = df[pd.to_numeric(df["batter"], errors="coerce").eq(ai_batter_id)].copy()
                    attack_prompt = build_ai_pitcher_batter_attack_prompt(
                        pitcher_name=selected_pitcher_name,
                        batter_name=ai_batter_name,
                        team_name=team_name,
                        dataset_source=dataset_source,
                        pitcher_df=df,
                        batter_df=batter_all_df,
                        direct_matchup_df=direct_matchup_df,
                    )
                    st.session_state[attack_key] = cached_ai_brief(
                        attack_prompt,
                        dataset_version(resolved_dataset_path),
                        AI_SCOUTING_MODEL_LABEL,
                        AI_MAX_INPUT_CHARS,
                        AI_MAX_OUTPUT_TOKENS,
                    )
                    st.caption(f"Attack-plan prompt sent {len(attack_prompt):,} characters.")
            except Exception as exc:
                st.error(f"Could not generate pitcher-batter attack plan: {exc}")
        if attack_key in st.session_state:
            st.markdown(st.session_state[attack_key])

    st.divider()
    st.markdown("**Team Matchup Brief**")

    if matchup_df.empty:
        st.info("No current roster matchup history is available, so an AI brief would be mostly caveats.")
    else:
        prompt = build_ai_brief_prompt(
            pitcher_name=selected_pitcher_name,
            team_name=team_name,
            dataset_source=dataset_source,
            data_start=data_start,
            data_end=data_end,
            pitcher_pitch_count=len(df),
            roster_count=len(scouting_roster),
            matchup_df=matchup_df,
            matchup_table=matchup_table,
        )
        st.caption(f"Team-brief prompt size: {len(prompt):,} characters.")
        ai_brief_key = (
            f"ai_brief:{resolved_dataset_path}:{dataset_version(resolved_dataset_path)}:"
            f"{pitching_team_name}:{selected_pitcher_id}:{team_name}:{roster_type}:{include_pitchers}:{ai_budget_signature}"
        )
        if st.button("Generate AI Brief", type="primary"):
            try:
                with st.spinner("Generating fast scouting brief..."):
                    st.session_state[ai_brief_key] = cached_ai_brief(
                        prompt,
                        dataset_version(resolved_dataset_path),
                        AI_SCOUTING_MODEL_LABEL,
                        AI_MAX_INPUT_CHARS,
                        AI_MAX_OUTPUT_TOKENS,
                    )
            except Exception as exc:
                st.error(f"Could not generate AI brief: {exc}")
        if ai_brief_key in st.session_state:
            st.markdown(st.session_state[ai_brief_key])

with tabs[4]:
    st.subheader("Dataset Status")
    st.write(f"Pitch source: `{dataset_source}`")
    st.write(f"Pitching team: `{pitching_team_name}`")
    st.write(f"Batting team: `{team_name}`")
    st.write(f"Roster scope: `{roster_label}`")
    st.write(f"Selected pitcher: `{selected_pitcher_name}` / MLBAM `{selected_pitcher_id}`")
    st.write(f"Local pitch file: `{resolved_dataset_path}`")
    st.write(f"Rows after pitcher filter: `{len(df):,}`")
    st.write(f"Total dataset rows: `{raw_dataset_rows:,}`")
    st.write(f"Unique batters faced: `{df['batter'].nunique():,}`")
    st.write(f"Unique plate appearances: `{df['pa_key'].nunique():,}`")
    st.code("python scripts/build_all_pitches_parquet.py --overwrite", language="bash")
