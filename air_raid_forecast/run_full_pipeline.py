"""
End-to-End Orchestrator for the Ukrainian Air Raid Defense Forecasting Pipeline.
Runs full data ingestion, spatial graph build, regularized ST-GNN training,
overfitting verification, and live inference engine initialization.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Robust repository path bootstrap
current_file_path = Path(__file__).resolve()
# Root directory is either parent of air_raid_forecast or the repository root
repo_root = current_file_path.parents[1] if current_file_path.parent.name == "air_raid_forecast" else current_file_path.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import time

def run_pipeline(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    val_days: int = 30,
    test_days: int = 30,
    sample_stride: int = 4
):
    start_time = time.time()
    print("=" * 80)
    print("       AIR RAID FORECAST PIPELINE - FULL PROCESSING ORCHESTRATOR       ")
    print("=" * 80)

    # Step 1: Ingest & Clean Data
    print("\n[Step 1/5] Checking and downloading official alerts dataset...")
    from air_raid_forecast.src.data.download_alerts import download_official_alerts, load_and_clean_alerts
    raw_path = download_official_alerts()
    df = load_and_clean_alerts(raw_path)
    print(f"  -> Successfully verified {len(df):,} cleaned records spanning {df['start'].min().strftime('%Y-%m-%d')} to {df['start'].max().strftime('%Y-%m-%d')}.")

    # Step 2: Build Spatial Graph & Raion Hierarchy
    print("\n[Step 2/5] Building geographic adjacency graph, Laplacian, and 136-raion hierarchy...")
    from air_raid_forecast.src.data.build_graph import build_adjacency_matrix, get_normalized_laplacian, REGIONS
    from air_raid_forecast.src.data.raion_hierarchy import build_and_save_raion_metadata, NUM_RAIONS
    adj = build_adjacency_matrix()
    lap = get_normalized_laplacian(adj)
    raion_meta_path = build_and_save_raion_metadata()
    print(f"  -> Generated {len(REGIONS)}x{len(REGIONS)} macro spatial matrix with {int(adj.sum()/2)} border connections.")
    print(f"  -> Generated {NUM_RAIONS}-node micro raion hierarchy & cluster assignment matrix S (136x25).")

    # Step 3: Run Anti-Leakage and Invariant Tests
    print("\n[Step 3/5] Executing automated anti-leakage, tensor integrity, and hierarchical tests...")
    import unittest
    tests_dir = os.path.join(str(repo_root), "air_raid_forecast", "tests")
    suite = unittest.defaultTestLoader.discover(tests_dir)
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)
    if not test_result.wasSuccessful():
        print("  -> ERROR: Anti-leakage or diagnostic unit tests failed! Aborting pipeline.")
        sys.exit(1)
    print("  -> All causality, anti-leakage, and hierarchical unit tests PASSED cleanly.")

    # Step 4: Hierarchical Multi-Scale Defense Training & Evaluation
    print("\n[Step 4/5] Executing Hierarchical ST-GNN Training (Macro 25-Oblast + Micro 136-Raion Isochrones)...")
    from air_raid_forecast.src.training.train_layered import train_and_evaluate_layered_system
    report_path = train_and_evaluate_layered_system(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        test_days=test_days
    )
    print(f"  -> Hierarchical defense training and evaluation complete. Report written to: {report_path}")

    # Step 5: Verify Live Inference Engine
    print("\n[Step 5/5] Testing production inference engine initialization and live simulation...")
    from air_raid_forecast.src.inference.predict_engine import AirRaidInferenceEngine
    best_weights = os.path.join(str(repo_root), "air_raid_forecast", "checkpoints", "layered_stgnn_best.pt")
    engine = AirRaidInferenceEngine(
        checkpoint_path=best_weights,
        device="cpu"
    )
    # Simulate active border alerts
    engine.update_live_state(["Сумська область", "Харківська область"])
    out = engine.predict_realtime(
        current_hour_utc=14.0,
        recent_telegram_posts=["Зафіксовано активність ворожих БпЛА на північному сході!"]
    )
    kyiv_fc = out['region_forecasts']['м. Київ']
    print(f"  -> Live engine initialized.")
    print(f"     Kyiv City Tactical Onset (1h):     {kyiv_fc['tactical_onset_prob_1h']}")
    print(f"     Kyiv City Corridor Incursion (4h): {kyiv_fc['operational_corridor_prob_4h']}")
    print(f"     Kyiv City 24h Expected Siren Time: {kyiv_fc['expected_daily_alert_hours_24h']} hours/day")
    print(f"     Kyiv City Time-to-Impact:          {kyiv_fc.get('time_to_impact_hours', 'N/A')} hours (tau={kyiv_fc.get('tau_hours', 'N/A')}h)")

    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"FULL PIPELINE EXECUTION FINISHED IN {total_duration/60:.2f} MINUTES")
    print(f"Artifacts generated:")
    print(f"  - Model checkpoint:      air_raid_forecast/checkpoints/layered_stgnn_best.pt")
    print(f"  - Hierarchical model:    air_raid_forecast/checkpoints/hierarchical_stgnn_best.pt")
    print(f"  - Layered report:        air_raid_forecast/checkpoints/layered_defense_report.json")
    print(f"  - Raion defense report:  air_raid_forecast/checkpoints/raion_defense_report.json")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Air Raid Forecast Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-days", type=int, default=30, help="Validation period in days")
    parser.add_argument("--test-days", type=int, default=30, help="Test holdout period in days")
    parser.add_argument("--sample-stride", type=int, default=4, help="Hourly subsample stride")
    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_days=args.val_days,
        test_days=args.test_days,
        sample_stride=args.sample_stride
    )
