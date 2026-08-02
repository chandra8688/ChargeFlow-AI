"""
ChargeFlow AI V2 — Phase 7 AI Decision & Recommendation Engine Demonstration
=============================================================================
Demonstrates real ML-backed decision logic across 3 naturally occurring cases:
  1. CASE A — STAY: Target station predicted occupancy < 0.70 policy threshold.
  2. CASE B — REROUTE: Target station predicted occupancy >= 0.70 and compatible candidate alternative offers >= 10% occupancy improvement.
  3. CASE C — NO_BETTER_ALTERNATIVE: Target station predicted occupancy >= 0.70, but best compatible candidate alternative offers < 10% occupancy improvement.
"""

import time
from src.services.forecast_service import ForecastService
from src.services.feature_service import FeatureService
from src.services.explainability_service import ExplainabilityService
from src.services.decision_service import DecisionService
from src.rag.llm_provider import MockLLMProvider
from src.rag.rag_service import RAGService


def run_phase7_demo():
    print("=" * 85)
    print("PHASE 7 — AI DECISION & RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 85)

    print("Initialising core services (ForecastService, FeatureService, ExplainabilityService) ...")
    forecast_svc = ForecastService(eager_load=True)
    feature_svc = FeatureService()
    explain_svc = ExplainabilityService(forecast_svc)

    rag_svc = RAGService(llm_provider=MockLLMProvider())
    rag_svc.initialize()

    dec_svc = DecisionService(
        forecast_service=forecast_svc,
        feature_service=feature_svc,
        explainability_service=explain_svc,
        rag_service=rag_svc,
    )

    demo_cases = [
        ("CASE A — STAY", "STA039", "2025-06-15 19:00:00", 28.0, False),
        ("CASE B — REROUTE", "STA001", "2025-06-15 19:00:00", 28.0, False),
        ("CASE C — NO_BETTER_ALTERNATIVE", "STA002", "2025-06-15 19:00:00", 28.0, False),
    ]

    for label, sid, ts, temp, hol in demo_cases:
        print("\n" + "-" * 85)
        print(f"QUERY: [{label}] Target Station: {sid} | Time: {ts} | Temp: {temp}°C")
        print("-" * 85)

        t0 = time.perf_counter()
        res = dec_svc.recommend(
            station_id=sid,
            prediction_time=ts,
            temperature_c=temp,
            is_holiday=hol,
            max_alternatives=3,
            include_rag_context=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        target = res["selected_station"]
        rec = res["recommendation"]
        reason = res["recommendation_reason"]
        top_alt = res["top_alternative"]

        print(f"  Target Station ID     : {target['station_id']} ({target['name']})")
        print(f"  Target City / Type    : {target['city']} | {target['charger_type']}")
        print(f"  ML Predicted Occupancy : {target['predicted_occupancy']*100:.1f}% ({target['status']})")
        print(f"  AI Policy Outcome     : >>> {rec} <<<")
        print(f"  Decision Reason       : {reason}")

        if top_alt:
            print(f"\n  Top Alternative Station:")
            print(f"    - ID / Name         : {top_alt['station_id']} ({top_alt['name']})")
            print(f"    - Charger Type      : {top_alt['charger_type']}")
            print(f"    - Predicted Occupancy: {top_alt['predicted_occupancy']*100:.1f}% ({top_alt['status']})")
            print(f"    - Haversine Distance: {top_alt['distance_km']:.2f} km")
            print(f"    - Occupancy Improvement: {top_alt['occupancy_improvement']*100:.1f}%")
        else:
            print("  Top Alternative Station: None")

        if res.get("tree_dispersion"):
            td = res["tree_dispersion"]
            print(f"\n  Model Estimator Dispersion (200 trees):")
            print(f"    - Tree Mean={td['tree_mean']:.4f} | Std={td['tree_std']:.4f} | Consensus={td['status_consensus_pct']:.1f}%")

        if res.get("rag_context"):
            rag = res["rag_context"]
            print(f"\n  Grounded Domain Advice (RAG):")
            print(f"    - Grounded Answer   : {rag['answer']}")

        print(f"  (Total Latency: {elapsed_ms:.1f} ms)")

    print("\n" + "=" * 85)
    print("PHASE 7 DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 85)


if __name__ == "__main__":
    run_phase7_demo()
