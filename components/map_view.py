"""
ChargeFlow AI — Map View Component
====================================
Interactive Plotly station map for the Streamlit dashboard.
The primary implementation lives in app.py::page_map().
This module provides reusable helper functions.
"""
import plotly.graph_objects as go
import pandas as pd

STATUS_COLORS = {
    "AVAILABLE": "#00D4AA",
    "MODERATE":  "#FFB347",
    "BUSY":      "#FF8C00",
    "CRITICAL":  "#FF4B6E",
}


def build_station_map(map_df: pd.DataFrame, zoom: float = 5.0) -> go.Figure:
    """
    Build a Plotly Scattermapbox figure for the given station DataFrame.

    Args:
        map_df : realtime_df (with latitude, longitude, status, etc.)
        zoom   : initial map zoom level

    Returns:
        Plotly Figure with carto-darkmatter tile style
    """
    map_df = map_df.copy()
    map_df["marker_size"] = (map_df["total_slots"] * 2.8 + 10).clip(12, 38)

    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        sub = map_df[map_df["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=sub["latitude"], lon=sub["longitude"],
            mode="markers", name=status,
            marker=go.scattermapbox.Marker(
                size=sub["marker_size"], color=color, opacity=0.88,
            ),
            text=sub.apply(lambda r:
                f"<b>{r['name']}</b><br>"
                f"{r['operator']}<br>"
                f"Free: {r['available_slots']}/{r['total_slots']}<br>"
                f"Wait: {r['estimated_wait_mins']:.1f} min<br>"
                f"Util: {r['utilization_pct']:.1f}%",
                axis=1),
            hovertemplate="%{text}<extra></extra>",
        ))

    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom),
        showlegend=True,
        legend=dict(orientation="v", x=0.01, y=0.99,
                    bgcolor="rgba(10,14,26,0.85)",
                    bordercolor="rgba(0,212,170,0.2)", borderwidth=1,
                    font=dict(color="#E2E8F0", size=12)),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
