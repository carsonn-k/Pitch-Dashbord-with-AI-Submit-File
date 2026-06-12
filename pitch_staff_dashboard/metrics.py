from __future__ import annotations

import numpy as np
import pandas as pd


SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
    "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
TAKE_DESCRIPTIONS = {
    "called_strike",
    "ball",
    "blocked_ball",
    "pitchout",
    "intent_ball",
    "automatic_ball",
    "automatic_strike",
    "hit_by_pitch",
}
ON_BASE_EVENTS = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch"}
EXTRA_BASE_EVENTS = {"double", "triple", "home_run"}
SLUG_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
FOUL_DESCRIPTIONS = {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"}


def with_pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable pitch outcome flags used by scouting summaries."""
    out = df.copy()
    desc = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str)
    events = out.get("events", pd.Series("", index=out.index)).fillna("").astype(str)
    zone = pd.to_numeric(out.get("zone", pd.Series(np.nan, index=out.index)), errors="coerce")
    plate_x = pd.to_numeric(out.get("plate_x", pd.Series(np.nan, index=out.index)), errors="coerce")
    plate_z = pd.to_numeric(out.get("plate_z", pd.Series(np.nan, index=out.index)), errors="coerce")
    sz_top = pd.to_numeric(out.get("sz_top", pd.Series(3.5, index=out.index)), errors="coerce").fillna(3.5)
    sz_bot = pd.to_numeric(out.get("sz_bot", pd.Series(1.5, index=out.index)), errors="coerce").fillna(1.5)
    launch_speed = pd.to_numeric(out.get("launch_speed", pd.Series(np.nan, index=out.index)), errors="coerce")
    launch_angle = pd.to_numeric(out.get("launch_angle", pd.Series(np.nan, index=out.index)), errors="coerce")

    out["is_swing"] = desc.isin(SWING_DESCRIPTIONS)
    out["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)
    out["is_take"] = desc.isin(TAKE_DESCRIPTIONS)
    out["is_in_play"] = desc.str.startswith("hit_into_play")
    out["is_foul"] = desc.isin(FOUL_DESCRIPTIONS)
    out["is_hit"] = events.isin(HIT_EVENTS)
    out["is_zone"] = zone.between(1, 9) | (plate_x.abs().le(0.83) & plate_z.between(sz_bot, sz_top))
    out["is_chase"] = out["is_swing"] & ~out["is_zone"]
    out["is_hard_hit"] = launch_speed.ge(95)
    out["is_barrel_like"] = launch_speed.ge(98) & launch_angle.between(24, 34)
    out["bases_on_contact"] = events.map(SLUG_BASES).fillna(0)
    out["is_on_base_event"] = events.isin(ON_BASE_EVENTS)
    out["is_extra_base_event"] = events.isin(EXTRA_BASE_EVENTS)
    return out


def batter_pitch_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Return every pitch a selected batter has seen in scouting-friendly order."""
    if df.empty:
        return pd.DataFrame()
    out = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()
    cols = [
        "game_date",
        "pitcher_display",
        "pitcher",
        "inning",
        "inning_topbot",
        "at_bat_number",
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
        "pa_result",
        "savant_url",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def batter_zone_outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize what happened to a batter by Statcast zone."""
    flagged = with_pitch_flags(df)
    if flagged.empty:
        return pd.DataFrame()

    grouped = (
        flagged.groupby("zone", dropna=False)
        .agg(
            Pitches=("pitch_number", "count"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            Fouls=("is_foul", "sum"),
            Takes=("is_take", "sum"),
            BallsInPlay=("is_in_play", "sum"),
            Hits=("is_hit", "sum"),
            HardHit=("is_hard_hit", "sum"),
            BarrelLike=("is_barrel_like", "sum"),
            AvgEV=("launch_speed", "mean"),
            AvgXwOBA=("estimated_woba_using_speedangle", "mean"),
            SlugBases=("bases_on_contact", "mean"),
        )
        .reset_index()
    )
    grouped["Swing %"] = grouped["Swings"] / grouped["Pitches"].replace(0, np.nan)
    grouped["Whiff/Swing %"] = grouped["Whiffs"] / grouped["Swings"].replace(0, np.nan)
    grouped["Foul/Swing %"] = grouped["Fouls"] / grouped["Swings"].replace(0, np.nan)
    grouped["Hit/BIP %"] = grouped["Hits"] / grouped["BallsInPlay"].replace(0, np.nan)
    grouped["Hard-Hit/BIP %"] = grouped["HardHit"] / grouped["BallsInPlay"].replace(0, np.nan)
    return grouped.sort_values(["Pitches", "Hits", "Whiffs"], ascending=False)


def batter_zone_leaders(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = batter_zone_outcome_summary(df)
    if summary.empty:
        return {
            "hits": pd.DataFrame(),
            "whiffs": pd.DataFrame(),
            "fouls": pd.DataFrame(),
            "takes": pd.DataFrame(),
            "damage": pd.DataFrame(),
        }
    return {
        "hits": summary.sort_values(["Hits", "Hit/BIP %", "Pitches"], ascending=False),
        "whiffs": summary.sort_values(["Whiffs", "Whiff/Swing %", "Pitches"], ascending=False),
        "fouls": summary.sort_values(["Fouls", "Foul/Swing %", "Pitches"], ascending=False),
        "takes": summary.sort_values(["Takes", "Pitches"], ascending=False),
        "damage": summary.sort_values(["AvgXwOBA", "AvgEV", "HardHit"], ascending=False),
    }


def rate(numerator: pd.Series, denominator: pd.Series | None = None) -> float:
    if denominator is None:
        denominator = pd.Series(True, index=numerator.index)
    denom = denominator.sum()
    if denom == 0:
        return np.nan
    return numerator.sum() / denom


def summarize_result_rates(df: pd.DataFrame) -> pd.DataFrame:
    flagged = with_pitch_flags(df)
    if flagged.empty:
        return pd.DataFrame()
    summary = pd.DataFrame(
        {
            "Pitches": [len(flagged)],
            "Swings": [int(flagged["is_swing"].sum())],
            "Swing %": [rate(flagged["is_swing"])],
            "Whiff %": [rate(flagged["is_whiff"], flagged["is_swing"])],
            "Chase %": [rate(flagged["is_chase"], ~flagged["is_zone"])],
            "In Play": [int(flagged["is_in_play"].sum())],
            "Hard-Hit %": [rate(flagged["is_hard_hit"], flagged["is_in_play"])],
            "Avg EV": [flagged["launch_speed"].mean()],
            "Avg LA": [flagged["launch_angle"].mean()],
            "Avg xwOBA": [flagged["estimated_woba_using_speedangle"].mean()],
        }
    )
    return summary


def batter_matchup_table(df: pd.DataFrame) -> pd.DataFrame:
    flagged = with_pitch_flags(df)
    if flagged.empty:
        return pd.DataFrame()

    grouped = flagged.groupby(["batter", "batter_display"], dropna=False)
    rows = []
    for (batter_id, batter_name), group in grouped:
        pa_count = group["pa_key"].nunique() if "pa_key" in group else np.nan
        events = group.dropna(subset=["events"]) if "events" in group else pd.DataFrame()
        rows.append(
            {
                "Batter": batter_name,
                "Batter ID": batter_id,
                "PA": pa_count,
                "Pitches": len(group),
                "Pitchers Faced": group["pitcher_display"].nunique() if "pitcher_display" in group else np.nan,
                "Swing %": rate(group["is_swing"]),
                "Whiff %": rate(group["is_whiff"], group["is_swing"]),
                "Chase %": rate(group["is_chase"], ~group["is_zone"]),
                "Hard-Hit": int(group["is_hard_hit"].sum()),
                "Barrel-Like": int(group["is_barrel_like"].sum()),
                "Avg EV": group["launch_speed"].mean(),
                "Avg xwOBA": group["estimated_woba_using_speedangle"].mean(),
                "PA Results": ", ".join(events["events"].dropna().astype(str).unique()[:6]) if not events.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["PA", "Pitches"], ascending=False)


def pitch_usage_by_count(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    table = (
        df.groupby(["count", "pitch_type"], dropna=False)
        .size()
        .rename("Pitches")
        .reset_index()
        .sort_values(["count", "Pitches"], ascending=[True, False])
    )
    totals = table.groupby("count")["Pitches"].transform("sum")
    table["Usage %"] = table["Pitches"] / totals
    return table


def pitch_usage_by_handedness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    table = (
        df.groupby(["stand", "pitch_type"], dropna=False)
        .size()
        .rename("Pitches")
        .reset_index()
        .sort_values(["stand", "Pitches"], ascending=[True, False])
    )
    totals = table.groupby("stand")["Pitches"].transform("sum")
    table["Usage %"] = table["Pitches"] / totals
    return table


def pitch_result_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "description" not in df:
        return pd.DataFrame()
    table = df.groupby(["description"], dropna=False).size().rename("Pitches").reset_index()
    table["Share"] = table["Pitches"] / table["Pitches"].sum()
    return table.sort_values("Pitches", ascending=False)


def contact_quality_by_pitch(df: pd.DataFrame) -> pd.DataFrame:
    flagged = with_pitch_flags(df)
    if flagged.empty:
        return pd.DataFrame()
    rows = []
    for pitch_type, group in flagged.groupby("pitch_type", dropna=False):
        rows.append(
            {
                "Pitch Type": pitch_type,
                "BIP": int(group["is_in_play"].sum()),
                "Avg EV": group["launch_speed"].mean(),
                "Max EV": group["launch_speed"].max(),
                "Avg LA": group["launch_angle"].mean(),
                "Hard-Hit %": rate(group["is_hard_hit"], group["is_in_play"]),
                "Barrel-Like": int(group["is_barrel_like"].sum()),
                "Avg xwOBA": group["estimated_woba_using_speedangle"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["BIP", "Avg EV"], ascending=False)


def putaway_pitch_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    flagged = with_pitch_flags(df)
    two_strike = flagged[pd.to_numeric(flagged["strikes"], errors="coerce").eq(2)]
    if two_strike.empty:
        return pd.DataFrame()
    rows = []
    for pitch_type, group in two_strike.groupby("pitch_type", dropna=False):
        rows.append(
            {
                "Pitch Type": pitch_type,
                "Two-Strike Pitches": len(group),
                "Whiffs": int(group["is_whiff"].sum()),
                "Whiff %": rate(group["is_whiff"], group["is_swing"]),
                "Put-Away Events": int(group["events"].isin(["strikeout", "strikeout_double_play"]).sum()),
                "Avg xwOBA": group["estimated_woba_using_speedangle"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["Put-Away Events", "Whiff %"], ascending=False)


def first_pitch_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    first = df[pd.to_numeric(df["pitch_number"], errors="coerce").eq(1)]
    if first.empty:
        return pd.DataFrame()
    table = first.groupby(["batter_display", "pitch_type"], dropna=False).size().rename("First Pitches").reset_index()
    totals = table.groupby("batter_display")["First Pitches"].transform("sum")
    table["Usage %"] = table["First Pitches"] / totals
    return table.sort_values(["batter_display", "First Pitches"], ascending=[True, False])


def two_strike_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    two_strike = df[pd.to_numeric(df.get("strikes", pd.Series(dtype=float)), errors="coerce").eq(2)]
    if two_strike.empty:
        return pd.DataFrame()
    table = two_strike.groupby(["batter_display", "pitch_type"], dropna=False).size().rename("Two-Strike Pitches").reset_index()
    totals = table.groupby("batter_display")["Two-Strike Pitches"].transform("sum")
    table["Usage %"] = table["Two-Strike Pitches"] / totals
    return table.sort_values(["batter_display", "Two-Strike Pitches"], ascending=[True, False])


def times_through_order_split(df: pd.DataFrame) -> pd.DataFrame:
    if "batting_order" not in df.columns or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["batting_order"] = pd.to_numeric(out["batting_order"], errors="coerce")
    if out["batting_order"].isna().all():
        return pd.DataFrame()
    pa_order = (
        out.drop_duplicates(["game_pk", "batting_team", "pa_key", "batter"])
        .sort_values(["game_pk", "batting_team", "at_bat_number"])
        .copy()
    )
    pa_order["TTO"] = pa_order.groupby(["game_pk", "batting_team", "batting_order"]).cumcount() + 1
    tto_map = pa_order.set_index("pa_key")["TTO"].to_dict()
    out["TTO"] = out["pa_key"].map(tto_map)
    flagged = with_pitch_flags(out)
    rows = []
    for tto, group in flagged.groupby("TTO"):
        rows.append(
            {
                "Times Through Order": int(tto),
                "PA": group["pa_key"].nunique(),
                "Pitches": len(group),
                "Whiff %": rate(group["is_whiff"], group["is_swing"]),
                "Hard-Hit %": rate(group["is_hard_hit"], group["is_in_play"]),
                "Avg xwOBA": group["estimated_woba_using_speedangle"].mean(),
            }
        )
    return pd.DataFrame(rows)


def zone_grid(
    df: pd.DataFrame,
    metric: str,
    *,
    x_bins: int = 12,
    z_bins: int = 12,
    x_range: tuple[float, float] = (-2.0, 2.0),
    z_range: tuple[float, float] = (0.5, 4.5),
) -> pd.DataFrame:
    flagged = with_pitch_flags(df)
    if flagged.empty:
        return pd.DataFrame()
    out = flagged.dropna(subset=["plate_x", "plate_z"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["x_bin"] = pd.cut(out["plate_x"], bins=np.linspace(*x_range, x_bins + 1), include_lowest=True)
    out["z_bin"] = pd.cut(out["plate_z"], bins=np.linspace(*z_range, z_bins + 1), include_lowest=True)

    metric_map = {
        "Pitches": ("pitch_number", "count"),
        "Hits": ("is_hit", "sum"),
        "Whiffs": ("is_whiff", "sum"),
        "Fouls": ("is_foul", "sum"),
        "Takes": ("is_take", "sum"),
        "Balls In Play": ("is_in_play", "sum"),
        "Hard-Hit": ("is_hard_hit", "sum"),
        "Swing Rate": ("is_swing", "mean"),
        "Whiff Rate": ("is_whiff", "mean"),
        "Chase Rate": ("is_chase", "mean"),
        "Take Rate": ("is_take", "mean"),
        "Hard-Hit Rate": ("is_hard_hit", "mean"),
        "Avg EV": ("launch_speed", "mean"),
        "Avg xwOBA": ("estimated_woba_using_speedangle", "mean"),
        "Slug Bases": ("bases_on_contact", "mean"),
    }
    value_col, agg = metric_map.get(metric, ("pitch_number", "count"))
    grid = out.groupby(["z_bin", "x_bin"], observed=False)[value_col].agg(agg).reset_index(name="value")
    grid["x_mid"] = grid["x_bin"].apply(lambda interval: interval.mid if pd.notna(interval) else np.nan).astype(float)
    grid["z_mid"] = grid["z_bin"].apply(lambda interval: interval.mid if pd.notna(interval) else np.nan).astype(float)
    return grid


def batter_zone_profile(df: pd.DataFrame, batter_id: int | float | str | None = None) -> dict[str, pd.DataFrame]:
    flagged = with_pitch_flags(df)
    if batter_id is not None and "batter" in flagged:
        flagged = flagged[flagged["batter"].astype(str).eq(str(batter_id))]
    if flagged.empty:
        return {
            "power_zone": pd.DataFrame(),
            "contact_zone": pd.DataFrame(),
            "whiff_zone": pd.DataFrame(),
            "chase_zone": pd.DataFrame(),
            "take_zone": pd.DataFrame(),
            "damage_by_pitch_type": pd.DataFrame(),
        }

    by_zone = (
        flagged.groupby("zone", dropna=False)
        .agg(
            Pitches=("pitch_number", "count"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            Takes=("is_take", "sum"),
            InPlay=("is_in_play", "sum"),
            HardHit=("is_hard_hit", "sum"),
            BarrelLike=("is_barrel_like", "sum"),
            AvgEV=("launch_speed", "mean"),
            AvgXwOBA=("estimated_woba_using_speedangle", "mean"),
            SlugBases=("bases_on_contact", "mean"),
        )
        .reset_index()
    )
    by_zone["Swing %"] = by_zone["Swings"] / by_zone["Pitches"]
    by_zone["Whiff %"] = by_zone["Whiffs"] / by_zone["Swings"].replace(0, np.nan)
    by_zone["Take %"] = by_zone["Takes"] / by_zone["Pitches"]
    by_zone["Hard-Hit %"] = by_zone["HardHit"] / by_zone["InPlay"].replace(0, np.nan)

    out_of_zone = flagged[~flagged["is_zone"]]
    chase_zone = (
        out_of_zone.groupby("zone", dropna=False)
        .agg(Pitches=("pitch_number", "count"), Chases=("is_chase", "sum"), Takes=("is_take", "sum"))
        .reset_index()
    )
    if not chase_zone.empty:
        chase_zone["Chase %"] = chase_zone["Chases"] / chase_zone["Pitches"]
        chase_zone["Take %"] = chase_zone["Takes"] / chase_zone["Pitches"]

    damage = (
        flagged.groupby(["pitch_type", "zone"], dropna=False)
        .agg(
            Pitches=("pitch_number", "count"),
            AvgEV=("launch_speed", "mean"),
            HardHit=("is_hard_hit", "sum"),
            BarrelLike=("is_barrel_like", "sum"),
            AvgXwOBA=("estimated_woba_using_speedangle", "mean"),
            SlugBases=("bases_on_contact", "mean"),
        )
        .reset_index()
        .sort_values(["AvgXwOBA", "AvgEV"], ascending=False)
    )

    return {
        "power_zone": by_zone.sort_values(["AvgXwOBA", "AvgEV", "HardHit"], ascending=False),
        "contact_zone": by_zone.assign(ContactRate=by_zone["InPlay"] / by_zone["Swings"].replace(0, np.nan)).sort_values(
            ["ContactRate", "InPlay"], ascending=False
        ),
        "whiff_zone": by_zone.sort_values(["Whiff %", "Whiffs"], ascending=False),
        "chase_zone": chase_zone.sort_values(["Chase %", "Chases"], ascending=False) if not chase_zone.empty else chase_zone,
        "take_zone": by_zone.sort_values(["Take %", "Takes"], ascending=False),
        "damage_by_pitch_type": damage,
    }


def attack_plan_for_batter(df: pd.DataFrame, batter_id: int | float | str) -> str:
    flagged = with_pitch_flags(df[df["batter"].astype(str).eq(str(batter_id))])
    if flagged.empty:
        return "No pitches in current filters."

    by_pitch = (
        flagged.groupby("pitch_type", dropna=False)
        .agg(
            Pitches=("pitch_number", "count"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            Chase=("is_chase", "sum"),
            OutZone=("is_zone", lambda s: int((~s).sum())),
            AvgXwOBA=("estimated_woba_using_speedangle", "mean"),
            AvgEV=("launch_speed", "mean"),
        )
        .reset_index()
    )
    by_pitch["Whiff %"] = by_pitch["Whiffs"] / by_pitch["Swings"].replace(0, np.nan)
    by_pitch["Chase %"] = by_pitch["Chase"] / by_pitch["OutZone"].replace(0, np.nan)

    weakness = by_pitch.sort_values(["Whiff %", "Chase %"], ascending=False).head(1)
    damage = by_pitch.sort_values(["AvgXwOBA", "AvgEV"], ascending=False).head(1)

    pieces = []
    sample = int(flagged["pa_key"].nunique()) if "pa_key" in flagged else len(flagged)
    pieces.append(f"Sample: {sample} PA / {len(flagged)} pitches.")

    if not weakness.empty:
        row = weakness.iloc[0]
        pitch = row["pitch_type"] if pd.notna(row["pitch_type"]) else "secondary stuff"
        pieces.append(f"Lean into {pitch} when ahead; current whiff/chase indicators are the best in this sample.")

    if not damage.empty:
        row = damage.iloc[0]
        pitch = row["pitch_type"] if pd.notna(row["pitch_type"]) else "middle-zone mistakes"
        pieces.append(f"Avoid predictable {pitch} in damage counts; it carries the highest contact quality here.")

    chase = flagged[flagged["is_chase"]]
    if not chase.empty:
        common_chase_pitch = chase["pitch_type"].mode()
        if not common_chase_pitch.empty:
            pieces.append(f"Expand with {common_chase_pitch.iloc[0]} after showing strikes.")

    takes = flagged[flagged["is_take"]]
    if not takes.empty:
        take_zone = takes["zone"].mode()
        if not take_zone.empty and pd.notna(take_zone.iloc[0]):
            pieces.append(f"Can steal takes around zone {int(take_zone.iloc[0])} in this dataset.")

    return " ".join(pieces)


def attack_plan_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for batter_id, group in df.groupby("batter", dropna=False):
        rows.append(
            {
                "Batter": group["batter_display"].iloc[0],
                "Batter ID": batter_id,
                "PA": group["pa_key"].nunique() if "pa_key" in group else np.nan,
                "Plan": attack_plan_for_batter(df, batter_id),
            }
        )
    return pd.DataFrame(rows).sort_values(["PA", "Batter"], ascending=[False, True])
