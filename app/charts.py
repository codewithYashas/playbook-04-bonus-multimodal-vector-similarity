"""Plotly figures for the app, on transparent backgrounds so they follow the Streamlit theme.

In the clickable maps every selectable item is a point of trace 0, so a click maps back
to a row of the frame the figure was built from (:func:`selected_index`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from explore import NOISE, cluster_color

ORANGE = "#FF734A"
LAVENDER = "#9085E9"
GREY = "#8A817C"
GREEN = "#199E70"
RED = "#D55181"
TRANSPARENT = "rgba(0,0,0,0)"
HIDDEN_AXIS = {"showticklabels": False, "showgrid": False, "zeroline": False, "title": ""}


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        showlegend=False,
    )
    return fig


def _steady(opacity: float) -> dict[str, dict[str, object]]:
    """Keep every point's look when one is clicked (plotly would dim the rest)."""
    style = {"marker": {"opacity": opacity}}
    return {"selected": style, "unselected": style}


def map_figure(points: pd.DataFrame, *, active: str | None = None, height: int = 520) -> go.Figure:
    """A 2-D embedding map.

    Args:
        points: x, y, id, label, kind ("video"/"image"/"frame") and cluster per point.
        active: An id to ring in orange (the details dialog's item).
    """
    is_image = (points["kind"] == "image").to_numpy()
    colors = [cluster_color(int(c)) for c in points["cluster"]]
    fig = go.Figure(
        go.Scatter(
            x=points["x"],
            y=points["y"],
            mode="markers",
            marker={
                "size": np.where(is_image, 18, 11).tolist(),
                "symbol": np.where(is_image, "star", "circle").tolist(),
                "color": colors,
                "opacity": 0.9,
                "line": {"width": 1, "color": "white"},
            },
            text=points["label"],
            customdata=points["id"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            **_steady(0.9),
        )
    )
    images = points[is_image]
    if not images.empty:
        fig.add_trace(
            go.Scatter(
                x=images["x"],
                y=images["y"],
                mode="text",
                text=images["label"],
                textposition="top center",
                hoverinfo="skip",
            )
        )
    if active is not None:
        hit = points[points["id"].astype(str) == str(active)]
        if not hit.empty:
            fig.add_trace(
                go.Scatter(
                    x=hit["x"],
                    y=hit["y"],
                    mode="markers",
                    marker={"size": 26, "color": TRANSPARENT, "line": {"width": 3, "color": ORANGE}},
                    hoverinfo="skip",
                )
            )
    fig.update_layout(clickmode="event+select", dragmode="pan", xaxis=HIDDEN_AXIS, yaxis=HIDDEN_AXIS)
    return _style(fig, height)


def neighbour_bars(neighbours: pd.DataFrame, label_col: str = "label") -> go.Figure:
    """Ranked neighbours as horizontal similarity bars (nearest on top)."""
    data = neighbours.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["similarity"],
            y=data[label_col],
            orientation="h",
            marker={"color": [ORANGE if k == "image" else LAVENDER for k in data.get("media_type", [])]},
            text=[f"{s:.3f}" for s in data["similarity"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    lo = max(0.0, float(neighbours["similarity"].min()) - 0.05) if not neighbours.empty else 0
    fig.update_layout(xaxis={"range": [lo, 1.0], "title": "cosine similarity"}, yaxis={"title": ""})
    return _style(fig, 40 + 28 * len(neighbours))


def timeline_figure(frames: pd.DataFrame, threshold: float, best_index: int | None) -> go.Figure:
    """Similarity of each sampled frame to the reference, over the clip, with the threshold."""
    colors = [GREEN if s >= threshold else GREY for s in frames["similarity"]]
    fig = go.Figure(
        go.Scatter(
            x=frames["timestamp_s"],
            y=frames["similarity"],
            mode="lines+markers",
            line={"color": LAVENDER},
            marker={"size": 9, "color": colors},
            customdata=frames["frame_index"],
            hovertemplate="t=%{x:.2f}s (frame %{customdata}): %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line={"dash": "dash", "color": ORANGE}, annotation_text=f"threshold {threshold:.2f}")
    if best_index is not None:
        best = frames[frames["frame_index"] == best_index]
        fig.add_trace(
            go.Scatter(
                x=best["timestamp_s"],
                y=best["similarity"],
                mode="markers",
                marker={"size": 18, "color": TRANSPARENT, "line": {"width": 3, "color": ORANGE}},
                hoverinfo="skip",
            )
        )
    fig.update_layout(xaxis={"title": "seconds into the clip"}, yaxis={"title": "cosine similarity"})
    return _style(fig, 300)


def ranking_figure(best: pd.DataFrame, threshold: float) -> go.Figure:
    """Every video's best-frame similarity to one reference, sorted, coloured by pass/fail."""
    data = best.sort_values("similarity")
    fig = go.Figure(
        go.Bar(
            x=data["similarity"],
            y=data["video_id"],
            orientation="h",
            marker={"color": [GREEN if s >= threshold else RED for s in data["similarity"]]},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=threshold, line={"dash": "dash", "color": ORANGE})
    lo = max(0.0, float(data["similarity"].min()) - 0.05) if not data.empty else 0
    fig.update_layout(xaxis={"range": [lo, 1.0], "title": "best-frame cosine similarity"}, yaxis={"title": ""})
    return _style(fig, 60 + 20 * len(data))


def selected_index(event: object, curve: int = 0) -> int | None:
    """The row index of the first clicked point on trace ``curve``, or None."""
    selection = event.get("selection") if isinstance(event, Mapping) else None
    points = selection.get("points") if isinstance(selection, Mapping) else None
    for point in points if isinstance(points, list) else []:
        if isinstance(point, Mapping) and point.get("curve_number", 0) == curve:
            for key in ("point_index", "point_number", "pointIndex", "pointNumber"):
                if isinstance(point.get(key), int | float):
                    return int(point[key])
    return None


__all__ = ["NOISE", "map_figure", "neighbour_bars", "ranking_figure", "selected_index", "timeline_figure"]
