from __future__ import annotations

import altair as alt
import pandas as pd

from .metrics import zone_grid

ZONE_CHART_WIDTH = 420
ZONE_CHART_HEIGHT = 560


def _empty_chart(message: str = "No data") -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"message": [message]}))
        .mark_text(size=16, color="#555")
        .encode(text="message:N")
        .properties(height=260)
    )


def usage_bar(table: pd.DataFrame, x: str, color: str = "pitch_type", y: str = "Usage %", title: str = "") -> alt.Chart:
    if table.empty:
        return _empty_chart()

    tooltip = [alt.Tooltip(f"{x}:N", title=x), alt.Tooltip(f"{color}:N", title="Pitch")]
    if "Pitches" in table.columns:
        tooltip.append(alt.Tooltip("Pitches:Q", format=","))
    tooltip.append(alt.Tooltip(f"{y}:Q", format=".1%"))

    return (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title="", sort=None),
            y=alt.Y(f"{y}:Q", title=y, axis=alt.Axis(format="%") if y.endswith("%") else alt.Axis()),
            color=alt.Color(f"{color}:N", title="Pitch"),
            tooltip=tooltip,
        )
        .properties(title=title, height=320)
    )


def result_bar(table: pd.DataFrame) -> alt.Chart:
    if table.empty:
        return _empty_chart()

    plot_table = table.copy()
    plot_table["description"] = plot_table["description"].fillna("missing").astype(str)
    return (
        alt.Chart(plot_table)
        .mark_bar(color="#2f6f9f")
        .encode(
            x=alt.X("Pitches:Q", title="Pitches"),
            y=alt.Y("description:N", title="", sort="-x"),
            tooltip=[alt.Tooltip("description:N", title="Result"), alt.Tooltip("Pitches:Q", format=",")],
        )
        .properties(title="Pitch Result Summary", height=max(260, min(520, 30 * len(plot_table))))
    )


def _add_cell_bounds(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    x_values = sorted(out["x_mid"].dropna().unique())
    z_values = sorted(out["z_mid"].dropna().unique())
    x_width = (x_values[1] - x_values[0]) if len(x_values) > 1 else 0.33
    z_width = (z_values[1] - z_values[0]) if len(z_values) > 1 else 0.33
    out["x0"] = out["x_mid"] - x_width / 2
    out["x1"] = out["x_mid"] + x_width / 2
    out["z0"] = out["z_mid"] - z_width / 2
    out["z1"] = out["z_mid"] + z_width / 2
    return out


def _strike_zone_layer() -> alt.Chart:
    zone = pd.DataFrame({"x0": [-0.83], "x1": [0.83], "z0": [1.5], "z1": [3.5]})
    return (
        alt.Chart(zone)
        .mark_rect(fillOpacity=0, stroke="black", strokeWidth=2)
        .encode(x="x0:Q", x2="x1:Q", y="z0:Q", y2="z1:Q")
    )


def zone_heatmap(df: pd.DataFrame, metric: str) -> alt.Chart:
    grid = zone_grid(df, metric)
    if grid.empty:
        return _empty_chart()

    grid = _add_cell_bounds(grid)
    heat = (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("x0:Q", title="Horizontal Location (catcher view)", scale=alt.Scale(domain=[-2, 2])),
            x2="x1:Q",
            y=alt.Y("z0:Q", title="Vertical Location", scale=alt.Scale(domain=[0.5, 4.5])),
            y2="z1:Q",
            color=alt.Color("value:Q", title=metric, scale=alt.Scale(scheme="redyellowblue", reverse=True)),
            tooltip=[
                alt.Tooltip("x_mid:Q", title="plate_x", format=".2f"),
                alt.Tooltip("z_mid:Q", title="plate_z", format=".2f"),
                alt.Tooltip("value:Q", title=metric, format=".3f"),
            ],
        )
    )
    return (heat + _strike_zone_layer()).properties(
        title=f"Zone Heatmap: {metric}",
        width=ZONE_CHART_WIDTH,
        height=ZONE_CHART_HEIGHT,
    )


def pitch_location_scatter(df: pd.DataFrame, color: str = "pitch_type") -> alt.Chart:
    if df.empty:
        return _empty_chart()

    plot_df = df.dropna(subset=["plate_x", "plate_z"]).copy()
    if plot_df.empty:
        return _empty_chart("No pitch locations")

    tooltip_cols = ["pitch_number", "count", "pitch_type", "description", "events", "release_speed"]
    tooltip = [alt.Tooltip(f"{col}:N" if col in {"count", "pitch_type", "description", "events"} else f"{col}:Q") for col in tooltip_cols if col in plot_df]
    color_encoding = alt.Color(f"{color}:N", title=color) if color in plot_df.columns else alt.value("#2f6f9f")

    points = (
        alt.Chart(plot_df)
        .mark_circle(size=70, opacity=0.78)
        .encode(
            x=alt.X("plate_x:Q", title="Horizontal Location (catcher view)", scale=alt.Scale(domain=[-2, 2])),
            y=alt.Y("plate_z:Q", title="Vertical Location", scale=alt.Scale(domain=[0.5, 4.5])),
            color=color_encoding,
            tooltip=tooltip,
        )
    )
    return (points + _strike_zone_layer()).properties(
        title="Individual Pitch Locations",
        width=ZONE_CHART_WIDTH,
        height=ZONE_CHART_HEIGHT,
    )
