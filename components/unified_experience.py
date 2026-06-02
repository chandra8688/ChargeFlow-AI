"""
ChargeFlow AI — Unified Charging Experience
============================================
Cross-operator EV charging interoperability page.
Simulates: charger discovery · smart routing · unified payment.

Design: matches existing dark theme (#0A0E1A + teal/purple palette).
Self-contained — imported by app.py, no existing pages modified.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ── Provider Brand Palette ────────────────────────────────────────────────────
PROVIDERS = {
    "Tata Power EZ":  {"color": "#00A3E0", "icon": "🔵", "bg": "rgba(0,163,224,0.07)",  "border": "rgba(0,163,224,0.30)"},
    "ChargeZone":     {"color": "#FF6B35", "icon": "🟠", "bg": "rgba(255,107,53,0.07)",  "border": "rgba(255,107,53,0.30)"},
    "Statiq":         {"color": "#00D4AA", "icon": "🟢", "bg": "rgba(0,212,170,0.07)",   "border": "rgba(0,212,170,0.30)"},
    "BPCL EV":        {"color": "#7B61FF", "icon": "🟣", "bg": "rgba(123,97,255,0.07)",  "border": "rgba(123,97,255,0.30)"},
    "NTPC Vidyut":    {"color": "#FFB347", "icon": "🟡", "bg": "rgba(255,179,71,0.07)",  "border": "rgba(255,179,71,0.30)"},
    "Ather Grid":     {"color": "#E040FB", "icon": "🔴", "bg": "rgba(224,64,251,0.07)",  "border": "rgba(224,64,251,0.30)"},
}

STATUS_STYLE = {
    "AVAILABLE": {"label": "Available", "color": "#00D4AA", "dot": "🟢"},
    "BUSY":      {"label": "Busy",      "color": "#FFB347", "dot": "🟡"},
    "FULL":      {"label": "Full",      "color": "#FF4B6E", "dot": "🔴"},
    "RESERVED":  {"label": "Reserved",  "color": "#7B61FF", "dot": "🟣"},
}

VEHICLE_COMPAT = {
    "Tata Nexon EV":  ["Tata Power EZ", "ChargeZone", "Statiq", "BPCL EV", "NTPC Vidyut"],
    "MG ZS EV":       ["Tata Power EZ", "ChargeZone", "BPCL EV", "NTPC Vidyut"],
    "Ather 450X":     ["Ather Grid", "Statiq"],
    "Kia EV6":        ["ChargeZone", "BPCL EV"],
    "BYD Atto 3":     ["ChargeZone", "BPCL EV", "Tata Power EZ"],
}


# ── Mock Discovery Dataset ────────────────────────────────────────────────────
DISCOVERY_STATIONS = [
    {
        "id": "UCE_001", "name": "Koramangala Tata Hub",
        "provider": "Tata Power EZ", "distance_km": 0.8,
        "charger_type": "DC CCS2 (50 kW)", "slots_free": 3, "slots_total": 6,
        "wait_mins": 0, "cost_per_kwh": 14.5, "status": "AVAILABLE",
        "eta_mins": 4, "rating": 4.7,
        "amenities": ["☕ Cafe", "📶 WiFi", "🅿️ Free parking"],
        "lat": 12.9310, "lon": 77.6290, "recommended": True,
    },
    {
        "id": "UCE_002", "name": "HSR Layout ChargeZone",
        "provider": "ChargeZone", "distance_km": 1.4,
        "charger_type": "DC CCS2 (60 kW)", "slots_free": 1, "slots_total": 4,
        "wait_mins": 12, "cost_per_kwh": 16.0, "status": "BUSY",
        "eta_mins": 7, "rating": 4.3,
        "amenities": ["🅿️ Covered parking"],
        "lat": 12.9150, "lon": 77.6378, "recommended": False,
    },
    {
        "id": "UCE_003", "name": "Indiranagar Statiq Point",
        "provider": "Statiq", "distance_km": 2.1,
        "charger_type": "AC Type 2 (22 kW)", "slots_free": 0, "slots_total": 4,
        "wait_mins": 28, "cost_per_kwh": 11.5, "status": "FULL",
        "eta_mins": 11, "rating": 4.0,
        "amenities": ["🛒 Mall access"],
        "lat": 12.9784, "lon": 77.6408, "recommended": False,
    },
    {
        "id": "UCE_004", "name": "MG Road BPCL EV Hub",
        "provider": "BPCL EV", "distance_km": 2.8,
        "charger_type": "DC CCS2 (150 kW)", "slots_free": 2, "slots_total": 3,
        "wait_mins": 8, "cost_per_kwh": 18.0, "status": "AVAILABLE",
        "eta_mins": 14, "rating": 4.5,
        "amenities": ["🍽️ Food court", "🔧 Service bay"],
        "lat": 12.9745, "lon": 77.6081, "recommended": False,
    },
    {
        "id": "UCE_005", "name": "BTM NTPC Vidyut Station",
        "provider": "NTPC Vidyut", "distance_km": 3.2,
        "charger_type": "AC Type 2 (7.4 kW)", "slots_free": 4, "slots_total": 4,
        "wait_mins": 0, "cost_per_kwh": 9.5, "status": "AVAILABLE",
        "eta_mins": 17, "rating": 3.9,
        "amenities": ["🅿️ Street parking"],
        "lat": 12.9165, "lon": 77.6101, "recommended": False,
    },
    {
        "id": "UCE_006", "name": "Sarjapur Ather Grid",
        "provider": "Ather Grid", "distance_km": 4.5,
        "charger_type": "AC Type 2 (7.4 kW)", "slots_free": 8, "slots_total": 8,
        "wait_mins": 0, "cost_per_kwh": 7.0, "status": "AVAILABLE",
        "eta_mins": 22, "rating": 4.1,
        "amenities": ["☕ Ather Café", "📶 WiFi"],
        "lat": 12.9080, "lon": 77.6852, "recommended": False,
    },
]


# ── Page Renderer ─────────────────────────────────────────────────────────────

def render_page(stations_df: pd.DataFrame, realtime_df: pd.DataFrame):
    """Main entry point — renders the full Unified Charging Experience page."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="page-title">⚡ Unified Charging Experience</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Cross-network charger discovery · AI smart routing · Unified access &amp; payment'
        ' — one interface for every provider'
        '</div>',
        unsafe_allow_html=True)

    # ── Interoperability Banner ───────────────────────────────────────────────
    n_avail = int((realtime_df["available_slots"] > 0).sum())
    n_total = len(realtime_df)
    st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,212,170,0.06),rgba(123,97,255,0.06));
            border:1px solid rgba(0,212,170,0.2);border-radius:14px;
            padding:14px 20px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <span style="font-size:26px;">🔗</span>
    <div style="flex:1;min-width:200px;">
      <div style="font-size:13px;font-weight:700;color:#00D4AA;letter-spacing:0.5px;">
        ChargeFlow AI Interoperability Layer &mdash; Active
      </div>
      <div style="font-size:11.5px;color:#8892A4;margin-top:3px;">
        {n_total} network stations monitored &nbsp;·&nbsp;
        {n_avail} available now &nbsp;·&nbsp;
        OCPP 2.0 compliant &nbsp;·&nbsp;
        Single account across all providers
      </div>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <span style="background:rgba(0,163,224,0.12);color:#00A3E0;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:600;">🔵 Tata Power</span>
      <span style="background:rgba(255,107,53,0.12);color:#FF6B35;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:600;">🟠 ChargeZone</span>
      <span style="background:rgba(0,212,170,0.12);color:#00D4AA;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:600;">🟢 Statiq</span>
      <span style="background:rgba(123,97,255,0.12);color:#7B61FF;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:600;">🟣 BPCL EV</span>
      <span style="background:rgba(255,179,71,0.12);color:#FFB347;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:600;">+2 networks</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── User Context Row ──────────────────────────────────────────────────────
    uc1, uc2, uc3, uc4 = st.columns(4)
    with uc1:
        location = st.selectbox(
            "📍 Your Location",
            ["Koramangala, Bengaluru", "Indiranagar, Bengaluru",
             "HSR Layout, Bengaluru", "Connaught Place, Delhi",
             "Bandra West, Mumbai"],
            key="uce_loc")
    with uc2:
        vehicle = st.selectbox(
            "🚗 Your Vehicle",
            ["Tata Nexon EV", "MG ZS EV", "Ather 450X", "Kia EV6", "BYD Atto 3"],
            key="uce_veh")
    with uc3:
        battery = st.slider("🔋 Battery (%)", 5, 80, 18, key="uce_bat")
    with uc4:
        target_charge = st.slider("🎯 Charge to (%)", 60, 100, 80, key="uce_tgt")

    energy_needed = max(0.5, (target_charge - battery) / 100 * 40.0)  # ~40 kWh pack

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 3-Column Layout ───────────────────────────────────────────────────────
    left, center, right = st.columns([2, 2.3, 1.7], gap="medium")

    recommended = next((s for s in DISCOVERY_STATIONS if s["recommended"]),
                       DISCOVERY_STATIONS[0])

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — Charger Discovery
    # ═════════════════════════════════════════════════════════════════════════
    with left:
        st.markdown('<div class="section-header">🔍 Nearby Chargers — All Networks</div>',
                    unsafe_allow_html=True)

        sort_by = st.radio(
            "Sort by:", ["Distance", "Availability", "Cost"],
            horizontal=True, key="uce_sort")

        compat_providers = VEHICLE_COMPAT.get(vehicle, list(PROVIDERS.keys()))

        order_map = {"AVAILABLE": 0, "BUSY": 1, "RESERVED": 2, "FULL": 3}
        if sort_by == "Distance":
            stations_sorted = sorted(DISCOVERY_STATIONS, key=lambda x: x["distance_km"])
        elif sort_by == "Availability":
            stations_sorted = sorted(DISCOVERY_STATIONS,
                                     key=lambda x: order_map.get(x["status"], 4))
        else:
            stations_sorted = sorted(DISCOVERY_STATIONS, key=lambda x: x["cost_per_kwh"])

        for sta in stations_sorted:
            prov = PROVIDERS.get(sta["provider"], PROVIDERS["Statiq"])
            scfg = STATUS_STYLE.get(sta["status"], STATUS_STYLE["BUSY"])
            is_compat = sta["provider"] in compat_providers

            compat_html = (
                "<span style='background:rgba(0,212,170,0.12);color:#00D4AA;"
                "border-radius:20px;padding:2px 9px;font-size:10px;'>&#10003; Compatible</span>"
                if is_compat else
                "<span style='background:rgba(255,75,110,0.12);color:#FF4B6E;"
                "border-radius:20px;padding:2px 9px;font-size:10px;'>&#10007; Incompatible</span>"
            )
            rec_glow = "box-shadow:0 0 18px rgba(0,212,170,0.22);" if sta["recommended"] else ""
            rec_html = (
                "<span style='background:#00D4AA;color:#0A0E1A;border-radius:20px;"
                "padding:2px 9px;font-size:10px;font-weight:700;margin-left:6px;'>&#11088; AI PICK</span>"
                if sta["recommended"] else ""
            )
            amenity_str = " &nbsp;&middot;&nbsp; ".join(sta["amenities"][:2])
            slots_color = ("#00D4AA" if sta["slots_free"] / max(sta["slots_total"],1) > 0.4
                           else "#FFB347" if sta["slots_free"] > 0 else "#FF4B6E")

            st.markdown(f"""
<div style="background:{prov['bg']};border:1px solid {prov['border']};border-radius:12px;
            padding:14px 16px;margin-bottom:10px;{rec_glow}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
    <div>
      <span style="font-size:13px;font-weight:700;color:#E2E8F0;">{sta['name']}</span>{rec_html}
      <div style="font-size:11px;color:{prov['color']};font-weight:600;margin-top:2px;">
        {prov['icon']} {sta['provider']}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:14px;font-weight:800;color:{scfg['color']};">
        {scfg['dot']} {scfg['label']}
      </div>
      <div style="font-size:11px;color:{slots_color};">{sta['slots_free']}/{sta['slots_total']} free</div>
    </div>
  </div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px;">
    <span style="background:rgba(255,255,255,0.05);color:#8892A4;border-radius:16px;padding:2px 8px;font-size:10px;">&#128205; {sta['distance_km']} km</span>
    <span style="background:rgba(255,255,255,0.05);color:#8892A4;border-radius:16px;padding:2px 8px;font-size:10px;">&#9889; {sta['charger_type']}</span>
    <span style="background:rgba(255,255,255,0.05);color:#8892A4;border-radius:16px;padding:2px 8px;font-size:10px;">&#8987; {sta['wait_mins']} min wait</span>
    <span style="background:rgba(255,255,255,0.05);color:#8892A4;border-radius:16px;padding:2px 8px;font-size:10px;">&#8377;{sta['cost_per_kwh']}/kWh</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;">
    {compat_html}
    <span style="font-size:10px;color:#8892A4;">{amenity_str}</span>
    <span style="font-size:10px;color:#FFB347;">{'★' * int(sta['rating'])} {sta['rating']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # CENTER PANEL — Smart Routing
    # ═════════════════════════════════════════════════════════════════════════
    with center:
        st.markdown('<div class="section-header">🧠 AI Smart Routing</div>',
                    unsafe_allow_html=True)

        est_charge_min = round(energy_needed / 50.0 * 60, 0)
        total_cost_est = round(energy_needed * recommended["cost_per_kwh"] + 2.0, 1)
        arrive_time = datetime.now().replace(second=0, microsecond=0)

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0A2B22,#0D1A2B);
            border:2px solid #00D4AA;border-radius:14px;padding:18px 20px;
            margin-bottom:14px;box-shadow:0 0 24px rgba(0,212,170,0.14);">
  <div style="font-size:10px;color:#00D4AA;letter-spacing:2px;font-weight:700;margin-bottom:8px;">
    &#11088; AI RECOMMENDED STATION
  </div>
  <div style="font-size:17px;font-weight:900;color:#E2E8F0;margin-bottom:3px;">
    {recommended['name']}
  </div>
  <div style="font-size:12px;color:#00A3E0;font-weight:600;margin-bottom:14px;">
    {PROVIDERS[recommended['provider']]['icon']} {recommended['provider']}
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:14px;">
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px;text-align:center;">
      <div style="font-size:22px;font-weight:900;color:#00D4AA;">{recommended['eta_mins']} min</div>
      <div style="font-size:9.5px;color:#8892A4;letter-spacing:1px;">ETA TO STATION</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px;text-align:center;">
      <div style="font-size:22px;font-weight:900;color:#FFB347;">{recommended['wait_mins']} min</div>
      <div style="font-size:9.5px;color:#8892A4;letter-spacing:1px;">QUEUE WAIT</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px;text-align:center;">
      <div style="font-size:22px;font-weight:900;color:#7B61FF;">{est_charge_min:.0f} min</div>
      <div style="font-size:9.5px;color:#8892A4;letter-spacing:1px;">CHARGE TIME</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px;text-align:center;">
      <div style="font-size:22px;font-weight:900;color:#E2E8F0;">&#8377;{total_cost_est}</div>
      <div style="font-size:9.5px;color:#8892A4;letter-spacing:1px;">EST. COST</div>
    </div>
  </div>

  <div style="background:rgba(0,212,170,0.06);border-left:3px solid #00D4AA;
              border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:#8892A4;
              line-height:1.6;">
    &#128161; <b style='color:#E2E8F0;'>Why this station?</b><br/>
    {recommended['slots_free']} slots open now &nbsp;&middot;&nbsp;
    Nearest DC fast charger &nbsp;&middot;&nbsp;
    Lowest congestion vs {sum(1 for s in DISCOVERY_STATIONS if s['status']!='AVAILABLE')} alternatives &nbsp;&middot;&nbsp;
    {' &middot; '.join(recommended['amenities'][:2])}
  </div>
</div>
""", unsafe_allow_html=True)

        # Route map
        u_lat, u_lon = 12.9279, 77.6271
        s_lat, s_lon = recommended["lat"], recommended["lon"]

        fig_route = go.Figure()
        # Dashed route line
        fig_route.add_trace(go.Scattermapbox(
            lat=[u_lat, s_lat], lon=[u_lon, s_lon],
            mode="lines",
            line=dict(width=3, color="#00D4AA"),
            showlegend=False,
        ))
        # Other stations
        for sta in DISCOVERY_STATIONS:
            if not sta["recommended"]:
                sc = STATUS_STYLE.get(sta["status"], STATUS_STYLE["BUSY"])
                fig_route.add_trace(go.Scattermapbox(
                    lat=[sta["lat"]], lon=[sta["lon"]],
                    mode="markers",
                    marker=dict(size=9, color=sc["color"], opacity=0.65),
                    showlegend=False,
                ))
        # User marker
        fig_route.add_trace(go.Scattermapbox(
            lat=[u_lat], lon=[u_lon], mode="markers+text",
            marker=dict(size=14, color="#7B61FF"),
            text=["&#128205; You"], textposition="top center",
            textfont=dict(color="#E2E8F0", size=11),
            showlegend=False,
        ))
        # Station marker
        fig_route.add_trace(go.Scattermapbox(
            lat=[s_lat], lon=[s_lon], mode="markers+text",
            marker=dict(size=18, color="#00D4AA"),
            text=["&#9889; " + recommended["name"][:18]],
            textposition="top right",
            textfont=dict(color="#E2E8F0", size=11),
            showlegend=False,
        ))
        fig_route.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=(u_lat+s_lat)/2, lon=(u_lon+s_lon)/2),
                        zoom=12.2),
            height=220, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_route, use_container_width=True,
                        config={"displayModeBar": False})

        # Battery optimization card
        st.markdown(f"""
<div style="background:rgba(123,97,255,0.06);border:1px solid rgba(123,97,255,0.2);
            border-radius:10px;padding:12px 16px;margin-top:10px;">
  <div style="font-size:11.5px;font-weight:700;color:#7B61FF;margin-bottom:6px;">
    &#128267; Battery Optimization Plan
  </div>
  <div style="font-size:11.5px;color:#8892A4;line-height:2.0;">
    Current: <b style='color:#E2E8F0;'>{battery}%</b>
    &rarr; Target: <b style='color:#E2E8F0;'>{target_charge}%</b><br/>
    Energy needed: <b style='color:#E2E8F0;'>{energy_needed:.1f} kWh</b><br/>
    Optimal: DC fast charge at 50 kW for your current SoC<br/>
    Avoid: overnight AC slow charge below 20% SoC
  </div>
</div>
""", unsafe_allow_html=True)

        # Traffic advisory
        st.markdown("""
<div style="background:rgba(255,179,71,0.06);border:1px solid rgba(255,179,71,0.2);
            border-radius:10px;padding:10px 14px;margin-top:10px;">
  <div style="font-size:11px;color:#FFB347;font-weight:600;">
    &#9888; Traffic Advisory
  </div>
  <div style="font-size:11px;color:#8892A4;margin-top:4px;line-height:1.7;">
    Moderate congestion on 80 Feet Road.
    Route avoids evening signal peak.
    Estimated arrival: <b style='color:#E2E8F0;'>7:42 PM</b>
  </div>
</div>
""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Unified Payment State Machine
    # ═════════════════════════════════════════════════════════════════════════
    with right:
        st.markdown('<div class="section-header">💳 Unified Access &amp; Pay</div>',
                    unsafe_allow_html=True)

        # ── Unified Account Badge ─────────────────────────────────────────
        st.markdown("""
<div style="background:linear-gradient(135deg,rgba(0,212,170,0.08),rgba(123,97,255,0.08));
            border:1px solid rgba(0,212,170,0.22);border-radius:10px;
            padding:10px 14px;margin-bottom:14px;text-align:center;">
  <div style="font-size:10px;color:#00D4AA;font-weight:700;letter-spacing:1.5px;">
    &#128279; CHARGEFLOW UNIFIED ACCOUNT
  </div>
  <div style="font-size:12px;color:#E2E8F0;font-weight:600;margin-top:4px;">
    Works across all 6 networks
  </div>
  <div style="font-size:10px;color:#8892A4;margin-top:2px;">
    No separate apps &nbsp;&middot;&nbsp; Single KYC &nbsp;&middot;&nbsp; OCPP 2.0
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Initialize flow state ─────────────────────────────────────────
        if "uce_flow" not in st.session_state:
            st.session_state["uce_flow"] = "IDLE"

        flow = st.session_state["uce_flow"]

        # ── STATE: IDLE ───────────────────────────────────────────────────
        if flow == "IDLE":
            pay_method = st.radio(
                "&#128176; Payment Method",
                ["&#128241; UPI (GPay / PhonePe / BHIM)",
                 "&#128179; Credit / Debit Card",
                 "&#128739; FASTag Wallet",
                 "&#128081; ChargeFlow Wallet"],
                key="uce_pay",
            )
            session_fee = 2.0
            total_cost = round(energy_needed * recommended["cost_per_kwh"] + session_fee, 1)

            st.markdown(f"""
<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
            border-radius:10px;padding:14px;margin:12px 0;">
  <div style="font-size:10px;color:#8892A4;font-weight:600;letter-spacing:1.2px;margin-bottom:10px;">
    COST ESTIMATE
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:7px;font-size:12px;">
    <span style="color:#8892A4;">Energy ({energy_needed:.1f} kWh)</span>
    <span style="color:#E2E8F0;">&#8377;{energy_needed * recommended['cost_per_kwh']:.1f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:7px;font-size:12px;">
    <span style="color:#8892A4;">Session fee</span>
    <span style="color:#E2E8F0;">&#8377;{session_fee}</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:7px;font-size:12px;">
    <span style="color:#8892A4;">Platform fee</span>
    <span style="color:#00D4AA;">&#8377;0 (waived)</span>
  </div>
  <div style="height:1px;background:rgba(255,255,255,0.07);margin:9px 0;"></div>
  <div style="display:flex;justify-content:space-between;font-size:14px;font-weight:700;">
    <span style="color:#E2E8F0;">Total</span>
    <span style="color:#00D4AA;">&#8377;{total_cost}</span>
  </div>
  <div style="font-size:10px;color:#8892A4;margin-top:7px;text-align:center;">
    Pre-authorized &middot; Charged on session end
  </div>
</div>
""", unsafe_allow_html=True)

            if st.button("&#128197; Reserve Charging Slot",
                         use_container_width=True, key="uce_reserve_btn"):
                st.session_state["uce_flow"]    = "RESERVED"
                st.session_state["uce_payment"] = pay_method
                st.session_state["uce_cost"]    = total_cost
                st.rerun()

        # ── STATE: RESERVED ───────────────────────────────────────────────
        elif flow == "RESERVED":
            pay_label = st.session_state.get("uce_payment", "UPI")[:22]
            cost      = st.session_state.get("uce_cost", 0)

            st.markdown(f"""
<div style="background:rgba(0,212,170,0.07);border:2px solid #00D4AA;
            border-radius:12px;padding:16px;margin-bottom:12px;text-align:center;">
  <div style="font-size:28px;margin-bottom:6px;">&#9989;</div>
  <div style="font-size:14px;font-weight:700;color:#00D4AA;">Slot Reserved!</div>
  <div style="font-size:12px;color:#8892A4;margin-top:8px;line-height:1.8;">
    {recommended['name']}<br/>
    Slot valid for <b style='color:#FFB347;'>10 minutes</b><br/>
    {pay_label}
  </div>
</div>
<div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:14px;
            margin-bottom:12px;text-align:center;">
  <div style="font-size:32px;font-weight:900;color:#00D4AA;margin-bottom:3px;">Slot B-3</div>
  <div style="font-size:10px;color:#8892A4;letter-spacing:1.5px;">YOUR ASSIGNED SLOT</div>
  <div style="font-size:12px;color:#E2E8F0;margin-top:9px;">
    Pre-auth: &#8377;{cost}
  </div>
</div>
""", unsafe_allow_html=True)

            ca, cb = st.columns(2)
            with ca:
                if st.button("&#9889; Start Session",
                             use_container_width=True, key="uce_start_btn"):
                    st.session_state["uce_flow"]       = "CHARGING"
                    st.session_state["uce_start_time"] = datetime.now()
                    st.rerun()
            with cb:
                if st.button("&#10060; Cancel",
                             use_container_width=True, key="uce_cancel_btn"):
                    st.session_state["uce_flow"] = "IDLE"
                    st.rerun()

        # ── STATE: CHARGING ───────────────────────────────────────────────
        elif flow == "CHARGING":
            # Auto-refresh while charging
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=2000, key="uce_charge_refresh")
            except ImportError:
                st.button("&#128260; Refresh Progress", key="uce_manual_refresh")

            start_t   = st.session_state.get("uce_start_time", datetime.now())
            elapsed_s = (datetime.now() - start_t).total_seconds()

            # 120× acceleration: 1 real second = 2 simulated minutes of charging
            charge_rate_kw  = 50.0
            charged_kwh     = min(energy_needed, elapsed_s * charge_rate_kw / 30.0)
            progress_pct    = min(1.0, charged_kwh / max(energy_needed, 0.1))
            current_battery = battery + (target_charge - battery) * progress_pct
            cost_so_far     = round(charged_kwh * recommended["cost_per_kwh"] + 2.0, 2)
            bar_w           = int(progress_pct * 100)

            st.markdown(f"""
<div style="background:rgba(0,212,170,0.06);border:2px solid #00D4AA;
            border-radius:12px;padding:16px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <div style="width:9px;height:9px;border-radius:50%;background:#00D4AA;
                animation:pulse-dot 1.2s infinite;flex-shrink:0;"></div>
    <span style="font-size:12px;font-weight:700;color:#00D4AA;letter-spacing:1.5px;">
      CHARGING ACTIVE
    </span>
  </div>
  <div style="text-align:center;margin-bottom:14px;">
    <div style="font-size:44px;font-weight:900;color:#00D4AA;">{current_battery:.0f}%</div>
    <div style="font-size:9.5px;color:#8892A4;letter-spacing:1.5px;">BATTERY LEVEL</div>
  </div>
  <div style="background:rgba(255,255,255,0.06);border-radius:8px;
              height:9px;margin-bottom:14px;overflow:hidden;">
    <div style="background:linear-gradient(90deg,#00D4AA,#7B61FF);height:100%;
                width:{bar_w}%;border-radius:8px;"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;">
    <div style="text-align:center;">
      <div style="color:#E2E8F0;font-weight:700;">{charged_kwh:.2f} kWh</div>
      <div style="color:#8892A4;">Dispensed</div>
    </div>
    <div style="text-align:center;">
      <div style="color:#E2E8F0;font-weight:700;">&#8377;{cost_so_far}</div>
      <div style="color:#8892A4;">Cost so far</div>
    </div>
    <div style="text-align:center;">
      <div style="color:#E2E8F0;font-weight:700;">{charge_rate_kw:.0f} kW</div>
      <div style="color:#8892A4;">Charge rate</div>
    </div>
    <div style="text-align:center;">
      <div style="color:#E2E8F0;font-weight:700;">{elapsed_s/60:.1f} min</div>
      <div style="color:#8892A4;">Session time</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            if st.button("&#9646; Stop &amp; Pay",
                         use_container_width=True, key="uce_stop_btn"):
                st.session_state["uce_flow"]       = "COMPLETE"
                st.session_state["uce_final_cost"] = cost_so_far
                st.session_state["uce_final_kwh"]  = charged_kwh
                st.rerun()

            if progress_pct >= 1.0:
                st.session_state["uce_flow"]       = "COMPLETE"
                st.session_state["uce_final_cost"] = st.session_state.get("uce_cost", 0)
                st.session_state["uce_final_kwh"]  = energy_needed
                st.rerun()

        # ── STATE: COMPLETE ───────────────────────────────────────────────
        elif flow == "COMPLETE":
            final_cost = st.session_state.get("uce_final_cost", 0)
            final_kwh  = st.session_state.get("uce_final_kwh", 0)
            pay_label  = st.session_state.get("uce_payment", "UPI")[:20]
            co2_saved  = round(final_kwh * 0.82, 1)

            st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,212,170,0.09),rgba(123,97,255,0.09));
            border:2px solid #00D4AA;border-radius:14px;
            padding:20px 16px;margin-bottom:14px;text-align:center;">
  <div style="font-size:38px;margin-bottom:8px;">&#9889;&#9989;</div>
  <div style="font-size:16px;font-weight:900;color:#00D4AA;margin-bottom:4px;">
    Session Complete!
  </div>
  <div style="font-size:11.5px;color:#8892A4;margin-bottom:16px;">
    Payment processed automatically
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
    <div style="background:rgba(255,255,255,0.05);border-radius:9px;padding:10px;">
      <div style="font-size:20px;font-weight:800;color:#E2E8F0;">{final_kwh:.1f} kWh</div>
      <div style="font-size:10px;color:#8892A4;">Energy Charged</div>
    </div>
    <div style="background:rgba(255,255,255,0.05);border-radius:9px;padding:10px;">
      <div style="font-size:20px;font-weight:800;color:#00D4AA;">&#8377;{final_cost}</div>
      <div style="font-size:10px;color:#8892A4;">Total Charged</div>
    </div>
  </div>
  <div style="background:rgba(0,212,170,0.07);border-radius:9px;padding:12px;
              font-size:11px;color:#8892A4;text-align:left;line-height:2.1;">
    &#9989; Payment: {pay_label}<br/>
    &#9989; Receipt sent to registered mobile<br/>
    &#9989; Session logged across all networks<br/>
    &#127807; CO&#8322; saved: <b style='color:#00D4AA;'>{co2_saved} kg</b> this session
  </div>
</div>
""", unsafe_allow_html=True)

            if st.button("&#128260; New Session",
                         use_container_width=True, key="uce_new_btn"):
                for k in ["uce_flow","uce_payment","uce_cost",
                          "uce_start_time","uce_final_cost","uce_final_kwh"]:
                    st.session_state.pop(k, None)
                st.rerun()

        # ── Interoperability footer ───────────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:10px;color:#8892A4;font-weight:600;letter-spacing:1px;margin-bottom:8px;">
  &#128279; INTEROPERABILITY BENEFITS
</div>
<div style="font-size:11px;color:#8892A4;line-height:2.2;">
  &#10003; One account &middot; All 6 networks<br/>
  &#10003; Single KYC &middot; No re-registration<br/>
  &#10003; Unified monthly billing<br/>
  &#10003; OCPP 2.0 plug-and-charge<br/>
  &#10003; MoRTH open standard compliant<br/>
  &#10003; Auto charger compatibility check
</div>
""", unsafe_allow_html=True)
