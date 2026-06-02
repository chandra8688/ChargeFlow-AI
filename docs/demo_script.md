# ChargeFlow AI — 3-Minute Demo Script (Final)

## PRE-DEMO SETUP (Do This 10 Min Before)
1. `streamlit run app.py`  → confirm loads at localhost:8501
2. Navigate to Overview page  → confirm KPI cards visible
3. Open browser in full-screen (F11)
4. Set browser zoom to 90%
5. Have slides open in Presenter View on second monitor/screen

---

## THE SCRIPT

### [0:00 – 0:20] HOOK — MEMORIZE THIS WORD FOR WORD

> "It's 7:30 PM in Bengaluru. You're driving your Tata Nexon EV.
> Battery at 12%. You open your charging app — and see 'Station Full.'
> Next one: no data. Third one: wrong charger type.
> This isn't a rare failure. This is India's EV charging experience,
> every single evening peak. I want to show you how ChargeFlow AI
> changes that — starting right now."

---

### [0:20 – 0:35] PROBLEM ANCHOR (one breath)

> "India has 1.7 million EVs, 40 operators, 12,000 chargers —
> and only 23% utilization. Three in four charger slots sit empty
> while drivers queue at the three stations everyone knows about.
> The problem isn't hardware. It's a complete absence of intelligence."

---

### [0:35 – 0:55] OPEN APP — OVERVIEW PAGE

*[Screen: Overview Dashboard, all KPI cards visible]*

> "This is our live prototype — 50 stations, five Indian cities,
> eight operators. The network KPIs update in real time.
> Active sessions, utilization rate, critical stations flagged,
> total grid load. Everything a network operator actually needs
> to see — in one view."

*[Point briefly to the demand trend chart]*

> "This is historical session volume by hour.
> You can see the morning peak at 8 AM, the lunch spike, and —
> watch this — the massive 7 PM evening surge.
> Our AI knows this is coming. 24 hours in advance."

---

### [0:55 – 1:25] SWITCH TO LIVE MAP — ENABLE SIMULATION

*[Click: Live Station Map in sidebar]*
*[Enable: Live Simulation toggle]*

> "Now here's the station map. Green: available. Amber: moderate.
> Red: critical. Watch when I enable Live Simulation —"

*[Click toggle, pause 3 seconds while map updates]*

> "Our system is now simulating OCPP telemetry ticks every 4 seconds.
> Real charger networks send these events in the same format.
> See those red stations? The AI has flagged them for
> immediate operator action. Queue forming. Load at critical.
> Without this layer, operators find out after angry customers call."

*[Hover over a red station to show popup]*

> "Click any station — you see occupancy, wait time, current load —
> all the signals an operator needs to act."

---

### [1:25 – 1:55] SWITCH TO RECOMMENDER

*[Click: Smart Recommender in sidebar]*
*[Select: Koramangala Bengaluru, Tata Nexon EV, 18% battery]*
*[Click: Find Best Station]*

> "Now I'm that driver from the opening. Koramangala.
> Tata Nexon EV. 18% battery. I click Find Best Station.
> ChargeFlow AI scores every compatible station across
> five criteria — proximity, availability, wait time,
> amenities, and cost — and returns my top three in under a second."

*[Point to the score badges on each card]*

> "Distance. Slots available. Wait time. All visible.
> And here's what makes this different—"

*[Scroll down to radar chart]*

> "This radar chart shows you exactly why each station scored what it did.
> Fully transparent. Judges can audit it. Drivers can trust it.
> This is not a black box."

---

### [1:55 – 2:20] SWITCH TO AI PREDICTOR

*[Click: AI Demand Predictor]*
*[Select: Bengaluru, All Types, All Days — chart renders]*

> "For operators — this is our 24-hour demand forecast for Bengaluru.
> Random Forest model. 100 decision trees. Trained on 10,000 sessions.
> The shaded band is our confidence interval — from the 10th
> to 90th percentile across all trees.
> The amber shading marks morning, lunch, and evening peaks —
> predicted 24 hours in advance.
> That purple dashed line? That's right now."

*[Point to current time marker]*

> "An operator looking at this at 5 PM knows —
> in two hours, this station will be at 95% capacity.
> They can adjust pricing, alert drivers, pre-position staff.
> That's what intelligence actually looks like."

---

### [2:20 – 2:40] ARCHITECTURE + IMPACT (Slide 7 or verbal)

> "The entire system has five modular layers — physical infrastructure
> at the base, data aggregation, orchestration, our AI engine,
> and the user interfaces at the top. Three AI models run in parallel:
> the demand predictor, the wait estimator using M/M/c queueing theory,
> and the recommender. Each has a specific job. Each is explainable."

---

### [2:40 – 3:00] CLOSE — MEMORIZE THIS TOO

> "If ChargeFlow AI brings India's charger utilization from
> 23% to just 50% — not even our full target —
> that unlocks eight thousand crore rupees in charging revenue
> that currently sits idle as empty parking spaces.
>
> India's EV revolution doesn't need more chargers.
> It needs smarter ones.
>
> ChargeFlow AI is how you make that happen. Thank you."

---

## TIMING GUIDE

| Segment | Time | Cumulative |
|---|---|---|
| Hook | 0:20 | 0:20 |
| Problem anchor | 0:15 | 0:35 |
| Overview page | 0:20 | 0:55 |
| Live Map + Simulation | 0:30 | 1:25 |
| Recommender | 0:30 | 1:55 |
| AI Predictor | 0:25 | 2:20 |
| Architecture summary | 0:20 | 2:40 |
| Closing line | 0:20 | 3:00 |

---

## EMERGENCY FALLBACKS

**If the simulation toggle doesn't refresh:**
> "The simulation is running in the background — I'll show you the effect in the station table below where the utilization numbers are updating."

**If the recommender returns 0 results:**
> "Let me expand the search radius—" [move slider to 50 km]

**If the app crashes:**
> "While we restart — let me show you the architecture on this slide, which is actually where I want to spend more time."
> [Switch to PPT Slide 7, continue from memory]

**If judges interrupt with questions:**
> Accept gracefully: "Great question — let me finish this flow in 30 seconds and I'll answer that directly."
