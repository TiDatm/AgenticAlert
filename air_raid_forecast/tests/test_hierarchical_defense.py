"""
Comprehensive Unit & Invariant Tests for Hierarchical Raion Defense Architecture:
- 136-Raion <-> 25-Oblast Spatio-Temporal Hierarchy & Assignment Matrix S
- Micro Adjacency Graph A_micro (136 x 136) & Directed Transition Operators
- Hierarchical Alert Inheritance (local_raion | parent_oblast)
- Multi-District De-Saturation Rule & Frontline Isolation (Nikopol vs Dnipro)
- Kinematic Isochrone Horizons tau(d) & Continuous Time-to-Impact Modeling
- HierarchicalSTGNN Forward Pass, Cross-Scale Gates, and Loss Gradient Flow
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

from air_raid_forecast.src.data.build_graph import REGIONS, REGION_TO_IDX
from air_raid_forecast.src.data.raion_hierarchy import (
    NUM_RAIONS,
    NUM_OBLASTS,
    NUM_LEAF_NODES,
    RAION_SPEC,
    LEAF_SPEC,
    RAION_TO_IDX,
    LEAF_TO_IDX,
    IDX_TO_LEAF,
    HROMADA_TO_RAION,
    RAION_TO_HROMADAS,
    build_cluster_assignment_matrix,
    build_leaf_to_raion_matrix,
    build_leaf_to_oblast_matrix,
    build_raion_adjacency_matrix,
    build_raion_directed_transition_matrices,
    compute_raion_threat_distances,
    compute_leaf_threat_distances,
    get_capital_raion_indices,
    get_raions_by_oblast,
)
from air_raid_forecast.src.data.multi_res_dataset import build_hierarchical_time_grids
from air_raid_forecast.src.models.hierarchical_stgnn import HierarchicalSTGNN, HierarchicalDefenseLoss


class TestHierarchicalRaionDefense(unittest.TestCase):
    def setUp(self):
        self.num_macro = 25
        self.num_micro = 136
        self.batch_size = 4
        self.x_kin = torch.randn(self.batch_size, self.num_macro, 32, 16)
        self.x_camp = torch.randn(self.batch_size, self.num_macro, 24, 10)
        self.x_raion = torch.randn(self.batch_size, self.num_micro, 32, 8)

    def test_raion_counts_and_assignment_matrix(self):
        """Invariant: Exactly 136 raions, orthogonal membership S sum across rows == 1."""
        self.assertEqual(len(RAION_SPEC), self.num_micro)
        s_mat = build_cluster_assignment_matrix()
        self.assertEqual(s_mat.shape, (self.num_micro, self.num_macro))

        # Every raion belongs to exactly 1 oblast
        row_sums = s_mat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

        # Every oblast contains at least 1 raion
        col_sums = s_mat.sum(axis=0)
        self.assertTrue(np.all(col_sums >= 1.0))

    def test_raion_adjacency_and_directed_transitions(self):
        """Invariant: A_micro is symmetric, connected, and transition matrices are valid stochastic operators."""
        adj = build_raion_adjacency_matrix()
        self.assertEqual(adj.shape, (self.num_micro, self.num_micro))
        # Symmetry check
        np.testing.assert_allclose(adj, adj.T)
        # Degree check: every node has at least 1 neighbor
        degrees = adj.sum(axis=1)
        self.assertTrue(np.all(degrees >= 1.0))

        p_fwd, p_bwd = build_raion_directed_transition_matrices()
        self.assertEqual(p_fwd.shape, (self.num_micro, self.num_micro))
        self.assertEqual(p_bwd.shape, (self.num_micro, self.num_micro))
        # Row stochasticity check (with self-loops)
        np.testing.assert_allclose(p_fwd.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(p_bwd.sum(axis=1), 1.0, atol=1e-5)

    def test_kinematic_isochrones_geometry(self):
        """Invariant: Border raions have fast horizons (<= 1h), deep-rear western raions have long horizons (>= 3.5h)."""
        d_threats, tau_hours, tau_steps = compute_raion_threat_distances()
        self.assertEqual(len(tau_hours), self.num_micro)

        # Bounds check
        self.assertTrue(np.all(tau_hours >= 0.5))
        self.assertTrue(np.all(tau_hours <= 6.0))

        # Kharkiv Raion (border) vs Lviv / Uzhhorod (deep rear)
        kharkiv_idx = RAION_TO_IDX["Харківський район"]
        lviv_idx = RAION_TO_IDX["Львівський район"]
        uzhhorod_idx = RAION_TO_IDX["Ужгородський район"]

        self.assertLessEqual(tau_hours[kharkiv_idx], 1.0)
        self.assertGreaterEqual(tau_hours[lviv_idx], 3.5)
        self.assertGreaterEqual(tau_hours[uzhhorod_idx], 4.0)

    def test_alert_inheritance_logic(self):
        """Test hierarchical alert inheritance: local | parent_oblast."""
        # Create synthetic alerts dataframe
        records = [
            # Oblast-level alert for Lviv
            {
                "region_title": "Львівська область",
                "level": "oblast",
                "start": pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 11:00:00", tz="UTC"),
            },
            # Raion-level alert strictly for Brovary Raion in Kyiv Oblast
            {
                "region_title": "Київська область",
                "raion": "Броварський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            }
        ]
        df_synth = pd.DataFrame(records)
        t_idx, g_strat, g_front, g_raion = build_hierarchical_time_grids(df_synth, freq="15min")

        # Step during Lviv oblast alert (10:30)
        idx_lviv = t_idx.get_loc(pd.Timestamp("2024-01-01 10:30:00", tz="UTC"))
        # All Lviv raions should inherit the alert
        by_obl = get_raions_by_oblast()
        lviv_raions = by_obl["Львівська область"]
        for r_id in lviv_raions:
            self.assertEqual(g_raion[idx_lviv, r_id], 1.0, f"Raion {r_id} failed to inherit Lviv alert")

        # Step during Brovary raion alert (12:30)
        idx_brov = t_idx.get_loc(pd.Timestamp("2024-01-01 12:30:00", tz="UTC"))
        brovary_id = RAION_TO_IDX["Броварський район"]
        bila_tserkva_id = RAION_TO_IDX["Білоцерківський район"]

        self.assertEqual(g_raion[idx_brov, brovary_id], 1.0)
        # Bila Tserkva must remain idle!
        self.assertEqual(g_raion[idx_brov, bila_tserkva_id], 0.0)

    def test_multi_district_de_saturation_rule(self):
        """Test physical de-saturation: Nikopol alone does NOT trigger Dnipropetrovsk strategic alert."""
        records = [
            # Isolated frontline tube artillery in Nikopol
            {
                "region_title": "Дніпропетровська область",
                "raion": "Нікопольський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 02:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
            },
            # Multi-district strike in Dnipro (Pavlohrad + Synelnykove)
            {
                "region_title": "Дніпропетровська область",
                "raion": "Павлоградський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 07:00:00", tz="UTC"),
            },
            {
                "region_title": "Дніпропетровська область",
                "raion": "Синельниківський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 07:00:00", tz="UTC"),
            }
        ]
        df_synth = pd.DataFrame(records)
        t_idx, g_strat, g_front, g_raion = build_hierarchical_time_grids(df_synth, freq="15min")
        dnipro_obl_idx = REGION_TO_IDX["Дніпропетровська область"]
        nikopol_r_idx = RAION_TO_IDX["Нікопольський район"]

        # 1. At 02:30 (Nikopol alone active):
        t_nikopol = t_idx.get_loc(pd.Timestamp("2024-01-01 02:30:00", tz="UTC"))
        self.assertEqual(g_raion[t_nikopol, nikopol_r_idx], 1.0, "Nikopol local alert must be active")
        self.assertEqual(g_strat[t_nikopol, dnipro_obl_idx], 0.0, "Dnipropetrovsk strategic must NOT saturate for Nikopol alone!")
        self.assertGreater(g_front[t_nikopol, dnipro_obl_idx], 0.0, "Nikopol artillery must be tracked in grid_front")

        # 2. At 06:30 (Multi-district saturation: Pavlohrad + Synelnykove active):
        t_multi = t_idx.get_loc(pd.Timestamp("2024-01-01 06:30:00", tz="UTC"))
        self.assertEqual(g_strat[t_multi, dnipro_obl_idx], 1.0, "Dnipropetrovsk strategic MUST trigger for multi-district saturation!")

    def test_hierarchical_stgnn_forward_and_loss(self):
        """Test HierarchicalSTGNN forward pass and gradient flow through all heads."""
        model = HierarchicalSTGNN()
        p1, p4, p24, p_iso, p_tti = model(self.x_kin, self.x_camp, self.x_raion)

        # Shape verification
        self.assertEqual(p1.shape, (self.batch_size, self.num_macro))
        self.assertEqual(p4.shape, (self.batch_size, self.num_macro))
        self.assertEqual(p24.shape, (self.batch_size, self.num_macro))
        self.assertEqual(p_iso.shape, (self.batch_size, self.num_micro))
        self.assertEqual(p_tti.shape, (self.batch_size, self.num_micro))

        # Time-to-impact bounds check [0.0, 6.0]
        self.assertTrue(torch.all(p_tti >= 0.0))
        self.assertTrue(torch.all(p_tti <= 6.0))

        # Test Loss and Backward Gradient Flow
        criterion = HierarchicalDefenseLoss()
        y1 = torch.randint(0, 2, (self.batch_size, self.num_macro)).float()
        y4 = torch.randint(0, 2, (self.batch_size, self.num_macro)).float()
        y24 = torch.rand(self.batch_size, self.num_macro) * 12.0
        mask_m = torch.ones(self.batch_size, self.num_macro)

        y_iso = torch.randint(0, 2, (self.batch_size, self.num_micro)).float()
        y_tti = torch.rand(self.batch_size, self.num_micro) * 6.0
        mask_r = torch.ones(self.batch_size, self.num_micro)

        loss = criterion(p1, p4, p24, y1, y4, y24, mask_m, p_iso, p_tti, y_iso, y_tti, mask_r)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")

    def test_macro_only_compatibility(self):
        """Test backward compatibility when x_raion is None."""
        model = HierarchicalSTGNN()
        p1, p4, p24, p_iso, p_tti = model(self.x_kin, self.x_camp, x_raion=None)
        self.assertEqual(p1.shape, (self.batch_size, self.num_macro))
        self.assertIsNone(p_iso)
        self.assertIsNone(p_tti)

    def test_unknown_raion_fallback_and_name_normalization(self):
        """Test edge cases: missing 'region_title' (using 'oblast'), suffix stripped raion, and unknown raion fallback."""
        records = [
            # Uses 'oblast' instead of 'region_title', and raion name without 'район'
            {
                "oblast": "Полтавська область",
                "raion": "Лубенський",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 04:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 05:00:00", tz="UTC"),
            },
            # Unknown raion in live telemetry: should fallback to constituent raions of parent oblast
            {
                "region_title": "Черкаська область",
                "raion": "Невідомий районний сектор",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 08:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 09:00:00", tz="UTC"),
            }
        ]
        df_synth = pd.DataFrame(records)
        t_idx, g_strat, g_front, g_raion = build_hierarchical_time_grids(df_synth, freq="15min")

        # 1. Lubny raion was specified without 'район' suffix
        idx_lub = t_idx.get_loc(pd.Timestamp("2024-01-01 04:30:00", tz="UTC"))
        lubny_idx = RAION_TO_IDX["Лубенський район"]
        self.assertEqual(g_raion[idx_lub, lubny_idx], 1.0, "Raion without 'район' suffix must match successfully")

        # 2. Unknown raion fallback in Cherkasy Oblast
        idx_cherk = t_idx.get_loc(pd.Timestamp("2024-01-01 08:30:00", tz="UTC"))
        by_obl = get_raions_by_oblast()
        for r_id in by_obl["Черкаська область"]:
            self.assertEqual(g_raion[idx_cherk, r_id], 1.0, f"Constituent raion {r_id} must be activated via fallback")

    def test_inference_engine_hierarchical_raion_predictions(self):
        """Test production inference engine outputs all 136 raion forecasts and respects alert inheritance."""
        from air_raid_forecast.src.inference.predict_engine import AirRaidInferenceEngine

        engine = AirRaidInferenceEngine(device="cpu")
        # Ingest alert for Sumy Oblast and Nikopol Raion
        engine.update_live_state(
            current_active_regions=["Сумська область"],
            current_active_raions=["Нікопольський район"]
        )

        out = engine.predict_realtime(current_hour_utc=14.0)

        # Must return region_forecasts (25) and raion_forecasts (136)
        self.assertIn("region_forecasts", out)
        self.assertEqual(len(out["region_forecasts"]), self.num_macro)

        if isinstance(engine.model, HierarchicalSTGNN):
            self.assertIn("raion_forecasts", out)
            raion_fc = out["raion_forecasts"]
            self.assertEqual(len(raion_fc), self.num_micro)

            # Check individual raion structure
            for r_name in ["Нікопольський район", "Дніпровський район", "Сумський район", "Львівський район"]:
                self.assertIn(r_name, raion_fc)
                fc = raion_fc[r_name]
                self.assertIn("isochrone_onset_prob", fc)
                self.assertIn("time_to_impact_hours", fc)
                self.assertIn("tau_hours", fc)
                self.assertTrue(0.0 <= fc["isochrone_onset_prob"] <= 1.0)
                self.assertTrue(0.0 <= fc["time_to_impact_hours"] <= 6.0)
                self.assertTrue(0.5 <= fc["tau_hours"] <= 6.0)

    def test_leaf_hierarchy_nodes_and_matrices(self):
        """Invariant: Exactly 204 leaf nodes (133 hromadas + 71 raions), valid orthogonal assignment matrices."""
        self.assertEqual(NUM_LEAF_NODES, 204)
        self.assertEqual(len(LEAF_SPEC), 204)

        mat_raion = build_leaf_to_raion_matrix()
        self.assertEqual(mat_raion.shape, (204, self.num_micro))
        # Every leaf node maps to exactly one parent raion
        np.testing.assert_allclose(mat_raion.sum(axis=1), 1.0)

        mat_oblast = build_leaf_to_oblast_matrix()
        self.assertEqual(mat_oblast.shape, (204, self.num_macro))
        # Every leaf node maps to exactly one parent oblast
        np.testing.assert_allclose(mat_oblast.sum(axis=1), 1.0)

        # Leaf threat distances and isochrone bounds
        d_threats, tau_hours, tau_steps = compute_leaf_threat_distances()
        self.assertEqual(len(tau_hours), 204)
        self.assertTrue(np.all(tau_hours >= 0.5))
        self.assertTrue(np.all(tau_hours <= 6.0))

    def test_downward_hierarchical_propagation(self):
        """Test 3-level Downward Hierarchical Propagation:
        Alert_leaf = Alert_local_leaf | Alert_parent_raion | Alert_parent_oblast
        """
        records = [
            # 1. Macro Oblast Alert for Kharkiv Oblast
            {
                "region_title": "Харківська область",
                "level": "oblast",
                "start": pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 02:00:00", tz="UTC"),
            },
            # 2. Raion-level alert for Nikopol Raion
            {
                "region_title": "Дніпропетровська область",
                "raion": "Нікопольський район",
                "level": "raion",
                "start": pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 04:00:00", tz="UTC"),
            },
            # 3. Micro Hromada-level alert strictly for Chervonohryhorivka Hromada
            {
                "region_title": "Дніпропетровська область",
                "raion": "Нікопольський район",
                "hromada": "Червоногригорівська територіальна громада",
                "level": "hromada",
                "start": pd.Timestamp("2024-01-01 05:00:00", tz="UTC"),
                "end": pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
            }
        ]
        df_synth = pd.DataFrame(records)

        # 5-tuple return with return_leaf_grid=True
        t_idx, g_strat, g_front, g_raion, g_leaf = build_hierarchical_time_grids(
            df_synth, freq="15min", return_leaf_grid=True
        )

        self.assertEqual(g_leaf.shape[1], 204)
        self.assertEqual(g_raion.shape[1], 136)

        # Case 1 (01:30): Kharkiv Oblast alert propagates downward to all its constituent leaves
        t_kh = t_idx.get_loc(pd.Timestamp("2024-01-01 01:30:00", tz="UTC"))
        kharkiv_leaves = [i for i, leaf in enumerate(LEAF_SPEC) if leaf["parent_oblast"] == "Харківська область"]
        self.assertGreater(len(kharkiv_leaves), 0)
        for leaf_idx in kharkiv_leaves:
            self.assertEqual(g_leaf[t_kh, leaf_idx], 1.0, f"Kharkiv leaf {leaf_idx} failed to inherit oblast alert")

        # Case 2 (03:30): Nikopol Raion alert propagates downward to ALL its constituent child hromadas
        t_nik = t_idx.get_loc(pd.Timestamp("2024-01-01 03:30:00", tz="UTC"))
        nikopol_leaves = [i for i, leaf in enumerate(LEAF_SPEC) if leaf["parent_raion"] == "Нікопольський район"]
        self.assertGreater(len(nikopol_leaves), 0)
        for leaf_idx in nikopol_leaves:
            self.assertEqual(g_leaf[t_nik, leaf_idx], 1.0, f"Nikopol leaf {leaf_idx} failed to inherit raion alert")

        # Non-Nikopol leaf in Dnipropetrovsk oblast must remain calm!
        dnipro_city_leaf = LEAF_TO_IDX.get("м. Дніпро та Дніпровська територіальна громада")
        if dnipro_city_leaf is not None:
            self.assertEqual(g_leaf[t_nik, dnipro_city_leaf], 0.0)

        # Case 3 (05:30): Local hromada alert for Chervonohryhorivka activates ONLY that leaf
        t_cherv = t_idx.get_loc(pd.Timestamp("2024-01-01 05:30:00", tz="UTC"))
        cherv_idx = LEAF_TO_IDX["Червоногригорівська територіальна громада"]
        self.assertEqual(g_leaf[t_cherv, cherv_idx], 1.0, "Chervonohryhorivka must be active")

        # Sister hromada Myrivska in Nikopol Raion must remain calm!
        myrivska_idx = LEAF_TO_IDX.get("Мирівська територіальна громада")
        if myrivska_idx is not None:
            self.assertEqual(g_leaf[t_cherv, myrivska_idx], 0.0, "Sister hromada must remain calm under local alert")

        # Strict Downward Propagation: Parent raion Nikopol remains calm (no upward reflection of child hromada alert)
        nikopol_r_idx = RAION_TO_IDX["Нікопольський район"]
        self.assertEqual(g_raion[t_cherv, nikopol_r_idx], 0.0, "Parent raion must remain calm under strict downward propagation")

    def test_inference_engine_leaf_forecasts(self):
        """Test inference engine calculates separated leaf forecasts for all 204 nodes."""
        from air_raid_forecast.src.inference.predict_engine import AirRaidInferenceEngine

        engine = AirRaidInferenceEngine(device="cpu")
        engine.update_live_state(
            current_active_regions=["Полтавська область"],
            current_active_raions=["Нікопольський район"],
            current_active_hromadas=["м. Нікополь та Нікопольська територіальна громада"]
        )
        out = engine.predict_realtime(current_hour_utc=16.0)

        self.assertIn("leaf_forecasts", out)
        leaf_fc = out["leaf_forecasts"]
        self.assertEqual(len(leaf_fc), 204)

        self.assertIn("hromada_forecasts", out)
        hromada_fc = out["hromada_forecasts"]
        self.assertEqual(len(hromada_fc), 133)

        # Inspect leaf forecast fields
        ch_name = "м. Нікополь та Нікопольська територіальна громада"
        self.assertIn(ch_name, leaf_fc)
        self.assertIn(ch_name, hromada_fc)
        item = leaf_fc[ch_name]
        self.assertEqual(item["parent_raion"], "Нікопольський район")
        self.assertEqual(item["parent_oblast"], "Дніпропетровська область")
        self.assertEqual(item["node_type"], "hromada")
        self.assertTrue(0.0 <= item["isochrone_onset_prob"] <= 1.0)
        self.assertTrue(0.5 <= item["tau_hours"] <= 6.0)

    def test_show_map_raions_and_hromadas_integration(self):
        """Test show_map AirRaidDashboard loads 136 raion geometries and child hromadas breakdown."""
        from air_raid_forecast.show_map import load_data, AirRaidDashboard

        gdf, metrics, layered_report = load_data()
        dashboard = AirRaidDashboard(gdf, metrics, layered_report)

        self.assertIsNotNone(dashboard.gdf_raions)
        self.assertEqual(len(dashboard.gdf_raions), 136)

        # Test raion value retrieval
        val_pr = dashboard._get_raion_value("Нікопольський район")
        self.assertIsInstance(val_pr, float)
        self.assertTrue(0.0 <= val_pr <= 1.0)

        # Test tactical probability retrieval mode
        dashboard.state["metric"] = "Raion Tactical Probability"
        val_prob = dashboard._get_raion_value("Дніпровський район")
        self.assertIsInstance(val_prob, float)
        self.assertTrue(0.0 <= val_prob <= 1.0)

        # Test sidebar rendering containing child hromadas and probabilities
        dashboard.state["selected_raion"] = "Нікопольський район"
        dashboard.redraw_dashboard()
        self.assertGreater(len(dashboard.active_hromada_screen_pts), 0)

        # Test leaf value retrieval for child hromada
        h_name = "м. Нікополь та Нікопольська територіальна громада"
        val_leaf = dashboard._get_leaf_value(h_name)
        self.assertIsInstance(val_leaf, float)
        self.assertTrue(0.0 <= val_leaf <= 1.0)

        # Test selecting individual child hromada and redraw
        dashboard.state["selected_hromada"] = h_name
        dashboard.redraw_dashboard()

        # Test keyboard navigation event
        class DummyKeyEvent:
            def __init__(self, key):
                self.key = key
        dashboard._on_key_press(DummyKeyEvent("h"))
        self.assertIsNotNone(dashboard.state.get("selected_hromada"))
        dashboard._on_key_press(DummyKeyEvent("escape"))
        self.assertIsNone(dashboard.state.get("selected_hromada"))

    def test_separated_evaluation_breakdown(self):
        """Test raion defense report contains separated metrics for all 136 raions and 204 leaves."""
        import json
        r_path = _PROJECT_ROOT / "air_raid_forecast" / "checkpoints" / "raion_defense_report.json"
        if not r_path.exists():
            self.skipTest("raion_defense_report.json not found")

        with open(r_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("raions_breakdown", data)
        self.assertIn("leaves_breakdown", data)

        r_bd = data["raions_breakdown"]
        l_bd = data["leaves_breakdown"]

        # 1. Exactly 136 raions categorized as raion
        raions_only = [v for v in r_bd.values() if v.get("type") == "raion"]
        self.assertEqual(len(raions_only), 136)

        # 2. Every raion has separated isochrone_onset_prob
        for r_item in raions_only:
            self.assertIn("isochrone_onset_prob", r_item)
            self.assertTrue(0.0 <= r_item["isochrone_onset_prob"] <= 1.0)
            self.assertIn("isochrone_pr_auc", r_item)
            self.assertIn("base_rate", r_item)

        # 3. All 204 leaf nodes present and have separated probabilities and metrics
        self.assertEqual(len(l_bd), 204)
        for l_name, l_item in l_bd.items():
            self.assertIn("isochrone_onset_prob", l_item)
            self.assertIn("isochrone_pr_auc", l_item)
            self.assertIn("base_rate", l_item)
            self.assertTrue(0.0 <= l_item["isochrone_onset_prob"] <= 1.0)

        # 4. Verify sister hromadas have separated, non-cloned metrics
        nikopol = l_bd.get("м. Нікополь та Нікопольська територіальна громада")
        pokrov = l_bd.get("м. Покров та Покровська територіальна громада")
        if nikopol and pokrov:
            # Base rates and metrics are computed from distinct local series
            self.assertNotEqual(nikopol["base_rate"], pokrov["base_rate"])
            self.assertNotEqual(nikopol["isochrone_pr_auc"], pokrov["isochrone_pr_auc"])


if __name__ == "__main__":
    unittest.main()

