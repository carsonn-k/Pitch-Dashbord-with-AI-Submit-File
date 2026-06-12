from __future__ import annotations

import numpy as np
import pandas as pd


# These columns identify one Statcast plate appearance without collapsing any
# pitch-level fields. `at_bat_number` is Statcast's plate-appearance counter
# inside a game; pitcher, batter, inning, and top/bottom protect against rare
# data quirks and make the grouping explicit for staff review.
PA_GROUP_COLS = ["game_pk", "at_bat_number", "pitcher", "batter", "inning", "inning_topbot"]


PITCH_DETAIL_COLUMNS = [
    "pitch_number",
    "count",
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
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "savant_link_type",
    "savant_url",
]


def ensure_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PA_GROUP_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    if "pitch_number" not in out.columns:
        out["pitch_number"] = pd.NA
    if "events" not in out.columns:
        out["events"] = pd.NA
    if "description" not in out.columns:
        out["description"] = pd.NA
    if "balls" not in out.columns:
        out["balls"] = pd.NA
    if "strikes" not in out.columns:
        out["strikes"] = pd.NA
    if "count" not in out.columns:
        out["count"] = out.apply(
            lambda row: f"{int(row['balls'])}-{int(row['strikes'])}" if pd.notna(row["balls"]) and pd.notna(row["strikes"]) else "",
            axis=1,
        )
    return out


def make_pa_key(df: pd.DataFrame) -> pd.Series:
    keyed = ensure_sequence_columns(df)
    return keyed[PA_GROUP_COLS].astype("string").fillna("NA").agg("|".join, axis=1)


def prepare_pitch_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Sort pitch-level Statcast rows while preserving plate appearance sequence.

    The dashboard never aggregates before this step. It first builds a stable
    `pa_key` from game, at-bat, pitcher, batter, inning, and top/bottom context,
    then sorts pitches inside that key by `pitch_number` ascending. Every later
    summary can reference `pa_key`, while pitch detail views keep all rows.
    """
    out = ensure_sequence_columns(df)
    out = out.copy()
    out["_original_row"] = np.arange(len(out))
    out["pitch_number"] = pd.to_numeric(out["pitch_number"], errors="coerce")
    out["pa_key"] = make_pa_key(out)

    sort_cols = PA_GROUP_COLS + ["pitch_number", "_original_row"]
    out = out.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    out["pa_pitch_index"] = out.groupby("pa_key", dropna=False).cumcount() + 1

    final_pitch = out.groupby("pa_key", dropna=False).tail(1).set_index("pa_key")
    final_events = final_pitch["events"].replace("", pd.NA)
    final_desc = final_pitch["description"].replace("", pd.NA)
    result_map = final_events.combine_first(final_desc).to_dict()
    out["pa_result"] = out["pa_key"].map(result_map)
    out = out.drop(columns=["_original_row"])
    return out


def validate_pitch_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return ordering issues where pitch numbers are missing, duplicated, or non-increasing."""
    checked = ensure_sequence_columns(df)
    if "pa_key" not in checked.columns:
        checked = checked.copy()
        checked["pa_key"] = make_pa_key(checked)
    issues = []
    for pa_key, group in checked.groupby("pa_key", sort=False, dropna=False):
        numbers = pd.to_numeric(group["pitch_number"], errors="coerce")
        raw_numbers = numbers.tolist()
        if numbers.isna().any():
            issues.append({"pa_key": pa_key, "reason": "missing pitch_number", "pitch_numbers": raw_numbers})
            continue
        diffs = numbers.diff().dropna()
        if (diffs <= 0).any():
            issues.append({"pa_key": pa_key, "reason": "pitch_number not strictly increasing", "pitch_numbers": raw_numbers})
    return pd.DataFrame(issues)


def plate_appearance_summary(df: pd.DataFrame) -> pd.DataFrame:
    sequenced = prepare_pitch_sequences(df)
    if sequenced.empty:
        return pd.DataFrame()

    rows = []
    for pa_key, group in sequenced.groupby("pa_key", sort=False, dropna=False):
        pitches = []
        for _, row in group.iterrows():
            pitch_type = row.get("pitch_type") if pd.notna(row.get("pitch_type")) else "UNK"
            desc = row.get("description") if pd.notna(row.get("description")) else ""
            count = row.get("count", "")
            pitches.append(f"P{int(row['pitch_number']) if pd.notna(row['pitch_number']) else '?'} {count} {pitch_type} {desc}")
        first = group.iloc[0]
        rows.append(
            {
                "pa_key": pa_key,
                "Date": first.get("game_date"),
                "Game": first.get("game_pk"),
                "Inning": first.get("inning"),
                "Top/Bot": first.get("inning_topbot"),
                "Pitcher": first.get("pitcher_display", first.get("pitcher")),
                "Batter": first.get("batter_display", first.get("batter")),
                "PA Result": group["pa_result"].iloc[-1],
                "Pitch Count": len(group),
                "Sequence": " | ".join(pitches),
            }
        )
    return pd.DataFrame(rows)


def pitch_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_pitch_sequences(df)
    for col in PITCH_DETAIL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[PITCH_DETAIL_COLUMNS + ["pa_key", "pa_result", "pitcher_display", "batter_display"]]


def add_previous_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    sequenced = prepare_pitch_sequences(df)
    if sequenced.empty:
        return sequenced
    group = sequenced.groupby("pa_key", sort=False, dropna=False)
    sequenced["previous_pitch_type"] = group["pitch_type"].shift(1)
    sequenced["previous_description"] = group["description"].shift(1)
    sequenced["previous_count"] = group["count"].shift(1)
    sequenced["next_pitch_type"] = group["pitch_type"].shift(-1)
    sequenced["next_description"] = group["description"].shift(-1)
    return sequenced


def next_pitch_tendency_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency-model placeholder for next-pitch tendency after prior pitch context."""
    enriched = add_previous_pitch_features(df)
    enriched = enriched.dropna(subset=["previous_pitch_type", "next_pitch_type"])
    if enriched.empty:
        return pd.DataFrame()
    table = (
        enriched.groupby(["previous_pitch_type", "previous_count", "previous_description", "stand", "next_pitch_type"], dropna=False)
        .size()
        .rename("Occurrences")
        .reset_index()
    )
    totals = table.groupby(["previous_pitch_type", "previous_count", "previous_description", "stand"])["Occurrences"].transform("sum")
    table["Next Pitch %"] = table["Occurrences"] / totals
    return table.sort_values(["Occurrences", "Next Pitch %"], ascending=False)


def sequence_outcome_after_pitch(pa_df: pd.DataFrame, pitch_number: int | float | str) -> pd.DataFrame:
    sequenced = prepare_pitch_sequences(pa_df)
    selected = pd.to_numeric(pd.Series([pitch_number]), errors="coerce").iloc[0]
    if pd.isna(selected):
        return pd.DataFrame()
    after = sequenced[pd.to_numeric(sequenced["pitch_number"], errors="coerce").gt(selected)].copy()
    cols = ["pitch_number", "count", "pitch_type", "description", "events", "pa_result", "savant_url", "savant_link_type"]
    for col in cols:
        if col not in after.columns:
            after[col] = pd.NA
    return after[cols]
