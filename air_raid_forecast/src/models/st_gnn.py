"""
PyTorch Multi-Resolution Adaptive Spatio-Temporal Graph Neural Network (Adaptive ST-GNN).
Simultaneously forecasts 1h, 4h, and 24h air raid alert probabilities across 25 regions.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root and parent are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from air_raid_forecast.src.models.layers import AdaptiveGraphConv, TemporalCausalGRU


class MultiResSTGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int = 25,
        fast_in_dim: int = 2,      # [is_active, neighbor_active_ratio]
        slow_in_dim: int = 3,      # [is_active, sin_hour, cos_hour]
        fast_hidden_dim: int = 32,
        slow_hidden_dim: int = 32,
        gnn_hidden_dim: int = 64,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # 1. Temporal Causal Encoders
        self.fast_encoder = TemporalCausalGRU(input_dim=fast_in_dim, hidden_dim=fast_hidden_dim)
        self.slow_encoder = TemporalCausalGRU(input_dim=slow_in_dim, hidden_dim=slow_hidden_dim)

        # 2. Multi-Scale Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(fast_hidden_dim + slow_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, fast_hidden_dim + slow_hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(fast_hidden_dim + slow_hidden_dim, gnn_hidden_dim)

        # 3. Adaptive Spatial Graph Convolution Layers
        self.gcn1 = AdaptiveGraphConv(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim
        )
        self.gcn2 = AdaptiveGraphConv(
            in_features=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim
        )

        # 4. Multi-Task Projection Heads (1h, 4h, 24h)
        self.head_1h = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head_4h = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head_24h = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x_fast: torch.Tensor,         # [B, 25, 8, fast_in_dim]
        x_slow: torch.Tensor,         # [B, 25, 24, slow_in_dim]
        static_laplacian: torch.Tensor # [25, 25]
    ) -> torch.Tensor:
        """
        Forward pass producing raw prediction logits for all 3 horizons.
        Returns:
            logits: [B, 25, 3] (index 0: 1h, index 1: 4h, index 2: 24h)
        """
        # Temporal representations
        h_fast = self.fast_encoder(x_fast) # [B, 25, fast_hidden]
        h_slow = self.slow_encoder(x_slow) # [B, 25, slow_hidden]

        # Multi-scale gated fusion
        h_cat = torch.cat([h_fast, h_slow], dim=-1) # [B, 25, 64]
        gate = self.fusion_gate(h_cat)
        h_fused = self.fusion_proj(gate * h_cat)     # [B, 25, gnn_hidden]

        # Spatial diffusion
        h_spatial = self.gcn1(h_fused, static_laplacian)
        h_spatial = self.gcn2(h_spatial, static_laplacian)

        # Multi-task heads
        out_1h = self.head_1h(h_spatial)   # [B, 25, 1]
        out_4h = self.head_4h(h_spatial)   # [B, 25, 1]
        out_24h = self.head_24h(h_spatial) # [B, 25, 1]

        # Concatenate across horizons: [B, 25, 3]
        logits = torch.cat([out_1h, out_4h, out_24h], dim=-1)
        return logits


if __name__ == "__main__":
    b, n = 4, 25
    x_f = torch.randn(b, n, 8, 2)
    x_s = torch.randn(b, n, 24, 3)
    lap = torch.eye(n)
    model = MultiResSTGNN(num_nodes=n)
    out = model(x_f, x_s, lap)
    print(f"Output logits shape: {out.shape}")
