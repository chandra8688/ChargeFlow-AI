"""
ChargeFlow AI — Analytics Component
======================================
Reusable chart-building functions for the analytics dashboard.
Primary implementation is in app.py::page_analytics().
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def build_demand_heatmap(sessions_df: pd.DataFrame) -> go.Figure:
    """City × Hour session volume heatmap."""
    pivot = sessions_df.groupby(["city", "hour"])["session_id"].count().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=list(pivot.index),
        colorscale=[[0,"#0A0E1A"],[0.2,"#0D2B3E"],[0.5,"#00D4AA"],[0.75,"#FFB347"],[1,"#FF4B6E"]],
        showscale=True,
        colorbar=dict(title="Sessions", title_font_color="#E2E8F0", tickfont=dict(color="#E2E8F0")),
        hovertemplate="City: %{y}<br>Hour: %{x}<br>Sessions: %{z}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        height=280,
        margin=dict(l=12, r=12, t=40, b=12),
        title=dict(text="Session Volume — City × Hour", font=dict(size=14, color="#E2E8F0")),
    )
    return fig


def build_operator_leaderboard(sessions_df: pd.DataFrame, realtime_df: pd.DataFrame) -> pd.DataFrame:
    """Compute operator efficiency scores."""
    op_rev  = sessions_df.groupby("operator")["revenue_inr"].sum().reset_index()
    op_rev.columns = ["operator", "revenue_90d"]
    op_util = realtime_df.groupby("operator").agg(
        avg_util=("utilization_pct", "mean"),
        stations=("station_id", "count"),
        critical=("status", lambda x: (x == "CRITICAL").sum()),
    ).reset_index()
    op_df = op_util.merge(op_rev, on="operator", how="left").fillna(0)
    op_df["efficiency_score"] = (
        op_df["avg_util"] * 0.5 +
        (op_df["revenue_90d"] / max(op_df["revenue_90d"].max(), 1) * 100) * 0.4 +
        (1 - op_df["critical"] / op_df["stations"].clip(lower=1)) * 10
    ).round(1)
    return op_df.sort_values("efficiency_score", ascending=False).reset_index(drop=True)
