"""
Unit tests for LayeredSTGNN architecture, DirectedAdvectiveGCN,
AirDefenseInterceptionDecay, DilatedCausalWaveNet, and Tactical Text Ingestion.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.models.layered_stgnn import LayeredSTGNN, LayeredDefenseLoss
from air_raid_forecast.src.models.layers import (
    DilatedCausalWaveNet,
    DirectedAdvectiveGCN,
    AirDefenseInterceptionDecay,
)
from air_raid_forecast.src.data.text_features import (
    parse_tactical_text,
    extract_threat_corridor,
    parse_batch_messages,
    TACTICAL_FEATURE_NAMES,
)
from air_raid_forecast.src.data.telegram_dataset import (
    build_tactical_text_grid,
    synthesize_deterministic_tactical_grid,
)
from air_raid_forecast.src.inference.predict_engine import AirRaidInferenceEngine


class TestLayeredSTGNN(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 25
        self.batch_size = 4
        self.x_kin = torch.randn(self.batch_size, self.num_nodes, 32, 16)
        self.x_camp = torch.randn(self.batch_size, self.num_nodes, 24, 10)
        self.p_fwd = torch.rand(self.num_nodes, self.num_nodes)
        self.p_bwd = torch.rand(self.num_nodes, self.num_nodes)

    def test_wavenet_forward(self):
        wavenet = DilatedCausalWaveNet(in_channels=16, hidden_channels=32)
        out = wavenet(self.x_kin)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, 32))

    def test_directed_gcn_forward(self):
        gcn = DirectedAdvectiveGCN(in_features=64, out_features=64, num_nodes=self.num_nodes)
        h = torch.randn(self.batch_size, self.num_nodes, 64)
        out = gcn(h, self.p_fwd, self.p_bwd)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, 64))

    def test_air_defense_interception_decay(self):
        decay_layer = AirDefenseInterceptionDecay(in_features=64, hidden_dim=16)
        h = torch.randn(self.batch_size, self.num_nodes, 64)
        p_eff = decay_layer(h, self.p_fwd)
        self.assertEqual(p_eff.shape, (self.batch_size, self.num_nodes, self.num_nodes))
        # Effective forward matrix must be <= original unattenuated matrix (up to numerical float precision)
        self.assertTrue(torch.all(p_eff >= 0.0))
        self.assertTrue(torch.all(p_eff <= self.p_fwd.unsqueeze(0) + 1e-5))

    def test_layered_stgnn_forward_and_backward(self):
        model = LayeredSTGNN(
            num_nodes=self.num_nodes,
            kinetic_in_dim=16,
            campaign_in_dim=10,
            kinetic_hidden_dim=32,
            campaign_hidden_dim=32,
            gnn_hidden_dim=64
        )
        p1, p4, p24, p_tti = model(self.x_kin, self.x_camp, self.p_fwd, self.p_bwd)
        self.assertEqual(p1.shape, (self.batch_size, self.num_nodes))
        self.assertEqual(p4.shape, (self.batch_size, self.num_nodes))
        self.assertEqual(p24.shape, (self.batch_size, self.num_nodes))
        self.assertEqual(p_tti.shape, (self.batch_size, self.num_nodes))
        self.assertTrue(torch.all(p24 >= 0.0))
        self.assertTrue(torch.all(p24 <= 24.0))
        self.assertTrue(torch.all(p_tti >= 0.0))
        self.assertTrue(torch.all(p_tti <= 6.0))

        # Backward compatibility check: return_tti=False
        p1_leg, p4_leg, p24_leg = model(self.x_kin, self.x_camp, self.p_fwd, self.p_bwd, return_tti=False)
        self.assertEqual(p1_leg.shape, (self.batch_size, self.num_nodes))

        # Test loss and backward gradients with TTI survival loss
        criterion = LayeredDefenseLoss()
        y1 = torch.randint(0, 2, (self.batch_size, self.num_nodes)).float()
        y4 = torch.randint(0, 2, (self.batch_size, self.num_nodes)).float()
        y24 = torch.rand(self.batch_size, self.num_nodes) * 12.0
        y_tti = torch.rand(self.batch_size, self.num_nodes) * 6.0
        mask = torch.ones(self.batch_size, self.num_nodes)

        loss = criterion(p1, p4, p24, y1, y4, y24, mask, pred_tti=p_tti, target_tti=y_tti)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(loss.item(), 1.0) # Check loss is finite and well-scaled

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")

    def test_tactical_text_extraction(self):
        # 1. MiG-31K
        t_mig = "Увага! Зафіксовано зліт МіГ-31К з аеродрому Саваслейка. Ракетна небезпека по всій території України!"
        f_mig = parse_tactical_text(t_mig)
        self.assertEqual(f_mig["flag_mig31k"], 1.0)
        self.assertEqual(f_mig["flag_clearance"], 0.0)

        # 2. Ballistics
        t_bal = "Загроза застосування балістичного озброєння зі сходу! Харків, Суми - в укриття!"
        f_bal = parse_tactical_text(t_bal)
        self.assertEqual(f_bal["flag_ballistics"], 1.0)

        # 3. Strategic Aviation
        t_strat = "Зафіксовано виліт 6 бортів Ту-95МС з аеродрому Оленья!"
        f_strat = parse_tactical_text(t_strat)
        self.assertEqual(f_strat["flag_strat_aviation"], 1.0)

        # 4. Shahed UAV
        t_shahed = "Група ударних БпЛА типу Shahed з півдня курсом на Миколаївщину!"
        f_shahed = parse_tactical_text(t_shahed)
        self.assertEqual(f_shahed["flag_shahed_group"], 1.0)

        # 5. KAB Guided Bomb
        t_kab = "Пуски керованих авіабомб (КАБ) тактичною авіацією на Харківщині!"
        f_kab = parse_tactical_text(t_kab)
        self.assertEqual(f_kab["flag_kab_tactical"], 1.0)

        # 6. Clearance
        t_clear = "Відбій загрози по МіГ-31К. По областях відбій тривоги."
        f_clear = parse_tactical_text(t_clear)
        self.assertEqual(f_clear["flag_clearance"], 1.0)
        self.assertEqual(f_clear["flag_mig31k"], 0.0)

    def test_temporal_causality_text(self):
        """
        Verify that features at step t do not change if future messages are added at t+1.
        """
        times = pd.date_range("2024-01-01 00:00:00", periods=20, freq="15min", tz="UTC")
        dummy_strat = np.zeros((20, 25), dtype=np.float32)

        grid_1 = build_tactical_text_grid(times, grid_strat=dummy_strat)

        # Perturb future siren state after step 10
        dummy_strat_future = dummy_strat.copy()
        dummy_strat_future[12:, :] = 1.0
        grid_2 = build_tactical_text_grid(times, grid_strat=dummy_strat_future)

        # Assert no change in past steps 0 to 10
        np.testing.assert_array_equal(
            grid_1[:10, :, :],
            grid_2[:10, :, :],
            err_msg="LEAKAGE: Tactical text grid at step <= 10 changed when future events at t >= 12 were altered!"
        )

    def test_inference_engine_realtime(self):
        engine = AirRaidInferenceEngine(device="cpu")
        engine.update_live_state(["Сумська область", "Харківська область"])
        out = engine.predict_realtime(
            current_hour_utc=14.0,
            recent_telegram_posts=["Увага! Зафіксовано пуски балістики зі сходу!"]
        )
        self.assertIn("tactical_flags", out)
        self.assertIn("region_forecasts", out)
        self.assertEqual(len(out["region_forecasts"]), 25)
        kyiv = out["region_forecasts"]["м. Київ"]
        self.assertIn("tactical_onset_prob_1h", kyiv)
        self.assertIn("operational_corridor_prob_4h", kyiv)
        self.assertIn("expected_daily_alert_hours_24h", kyiv)
        self.assertIn("time_to_impact_hours", kyiv)
        self.assertIn("tau_hours", kyiv)
        self.assertIn("isochrone_horizon_hours", kyiv)
        self.assertTrue(0.0 <= kyiv["time_to_impact_hours"] <= 6.0)
        self.assertTrue(1.0 <= kyiv["tau_hours"] <= 4.5)

    def test_regional_threat_distances_and_isochrones(self):
        from air_raid_forecast.src.data.build_graph import (
            compute_regional_threat_distances,
            REGIONAL_ISOCHRONES,
            REGIONAL_ISOCHRONE_STEPS,
            REGIONS,
        )
        d_threats, tau_hours, tau_steps = compute_regional_threat_distances()
        self.assertEqual(len(d_threats), 25)
        self.assertEqual(len(tau_hours), 25)
        self.assertEqual(len(tau_steps), 25)

        valid_taus = {1.0, 2.0, 3.0, 4.5}
        valid_steps = {4, 8, 12, 18}
        for tau, step in zip(tau_hours, tau_steps):
            self.assertIn(tau, valid_taus)
            self.assertIn(step, valid_steps)
            self.assertEqual(int(round(tau * 4)), step)

        # Border vs Deep Rear
        kharkiv_idx = REGIONS.index("Харківська область")
        lviv_idx = REGIONS.index("Львівська область")
        self.assertEqual(tau_hours[kharkiv_idx], 1.0)
        self.assertEqual(tau_hours[lviv_idx], 4.5)

    def test_layered_dataset_regional_isochrone_targets(self):
        from air_raid_forecast.src.data.layered_dataset import LayeredDefenseDataset
        from air_raid_forecast.src.data.build_graph import build_adjacency_matrix
        adj = build_adjacency_matrix()
        grid = np.zeros((300, 25), dtype=np.float32)
        # Put an alert at step 105 for region 0 (which is within forecast window for t=96)
        grid[105:110, 0] = 1.0
        time_idx = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
        ds = LayeredDefenseDataset(grid, time_idx, adj)
        self.assertGreater(len(ds), 0)
        sample = ds.get_sample(0, return_regional_iso=True)
        # Should return 8 items: x_kin, x_camp, y1, y4, y24, mask, y_iso, y_tti
        self.assertEqual(len(sample), 8)
        y_iso = sample[6]
        y_tti = sample[7]
        self.assertEqual(y_iso.shape, (25,))
        self.assertEqual(y_tti.shape, (25,))
        self.assertTrue(np.all((y_iso == 0.0) | (y_iso == 1.0)))
        self.assertTrue(np.all((y_tti >= 0.0) & (y_tti <= 6.0)))

    def test_dashboard_metric_resolution(self):
        """
        Verify that dashboard metric extraction handles all metrics, horizons, and fallback cases
        without raising NameError or KeyError.
        """
        from air_raid_forecast.show_map import load_data, AirRaidDashboard
        gdf, metrics, layered_report = load_data()
        dashboard = AirRaidDashboard(gdf, metrics, layered_report)

        # Test Layered mode metrics across 1h, 4h, 24h
        dashboard.state["mode"] = "Layered Defense (De-biased)"
        for horizon in ["1h (Tactical)", "4h (Operational)", "24h (Strategic)"]:
            dashboard.state["horizon"] = horizon
            for metric in dashboard._get_metric_options():
                dashboard.state["metric"] = metric
                val_kyiv = dashboard._get_region_value("м. Київ")
                val_lviv = dashboard._get_region_value("Львівська область")
                self.assertFalse(np.isnan(val_kyiv))
                self.assertFalse(np.isnan(val_lviv))

        # Test fallback path when regional breakdown has missing keys
        dashboard.layered_report = {"regional_breakdown": {"м. Київ": {}}}
        dashboard.state["horizon"] = "1h (Tactical)"
        dashboard.state["metric"] = "Tactical Precision Lift"
        self.assertEqual(dashboard._get_region_value("м. Київ"), 10.0)

        dashboard.state["horizon"] = "4h (Operational)"
        dashboard.state["metric"] = "Corridor Precision Lift"
        self.assertEqual(dashboard._get_region_value("м. Київ"), 5.0)

    def test_build_dual_time_grids_desaturation(self):
        """
        Verify multi-district & capital-weighted de-saturation in build_dual_time_grids:
        1. Nikopol alone maps strictly to grid_front without saturating grid_strat.
        2. Capital district (Дніпровський район) sounds strategic grid_strat.
        3. Simultaneous multi-district alerts (>= 2 raions) sound strategic grid_strat.
        4. Official oblast alert sounds strategic grid_strat.
        """
        from air_raid_forecast.src.data.multi_res_dataset import build_dual_time_grids
        from air_raid_forecast.src.data.build_graph import REGION_TO_IDX

        records = [
            # 1. Nikopol frontline tube artillery alone (01:00 - 02:00)
            {
                "region_title": "Дніпропетровська область",
                "raion": "Нікопольський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 02:00:00", tz="UTC"),
            },
            # 2. Capital district alert (03:00 - 04:00)
            {
                "region_title": "Дніпропетровська область",
                "raion": "Дніпровський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 04:00:00", tz="UTC"),
            },
            # 3. Two independent non-capital raions simultaneously (05:00 - 06:00)
            {
                "region_title": "Дніпропетровська область",
                "raion": "Павлоградський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 05:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
            },
            {
                "region_title": "Дніпропетровська область",
                "raion": "Синельниківський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 05:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
            },
            # 4. Official oblast-wide alert (07:00 - 08:00)
            {
                "region_title": "Дніпропетровська область",
                "level": "oblast",
                "start": pd.Timestamp("2024-01-01 07:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 08:00:00", tz="UTC"),
            },
        ]
        df_synth = pd.DataFrame(records)
        t_idx, grid_strat, grid_front = build_dual_time_grids(df_synth, freq="15min")
        dnipro_idx = REGION_TO_IDX["Дніпропетровська область"]

        # Case 1: Nikopol alone -> grid_strat == 0.0, grid_front > 0.0
        t1 = t_idx.get_loc(pd.Timestamp("2024-01-01 01:30:00", tz="UTC"))
        self.assertEqual(grid_strat[t1, dnipro_idx], 0.0, "Nikopol alone must not saturate grid_strat")
        self.assertGreater(grid_front[t1, dnipro_idx], 0.0, "Nikopol alert must be captured in grid_front")

        # Case 2: Capital district alert -> grid_strat == 1.0
        t2 = t_idx.get_loc(pd.Timestamp("2024-01-01 03:30:00", tz="UTC"))
        self.assertEqual(grid_strat[t2, dnipro_idx], 1.0, "Capital district alert must trigger grid_strat")

        # Case 3: >= 2 raions alert -> grid_strat == 1.0
        t3 = t_idx.get_loc(pd.Timestamp("2024-01-01 05:30:00", tz="UTC"))
        self.assertEqual(grid_strat[t3, dnipro_idx], 1.0, "Multi-raion alert must trigger grid_strat")

        # Case 4: Oblast-wide alert -> grid_strat == 1.0
        t4 = t_idx.get_loc(pd.Timestamp("2024-01-01 07:30:00", tz="UTC"))
        self.assertEqual(grid_strat[t4, dnipro_idx], 1.0, "Oblast-wide alert must trigger grid_strat")


if __name__ == "__main__":
    unittest.main()

