"""
PyTorch Layers for Adaptive Spatio-Temporal Graph Neural Network:
- AdaptiveGraphConv: Blends static geographic adjacency with learnable latent embeddings
- CausalTemporalGRU: Causal unidirectional recurrent temporal processor
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """
    Graph Convolution combining static geographic Laplacian and learnable adaptive adjacency.
    H_out = ReLU( A_norm * H * W_geo + A_adapt * H * W_adapt + H * W_self )
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int = 25,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes

        # Trainable node embeddings for discovering latent corridors
        self.node_embed1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_embed2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_embed1)
        nn.init.xavier_uniform_(self.node_embed2)

        # Transformation weight matrices
        self.weight_geo = nn.Linear(in_features, out_features, bias=False)
        self.weight_adapt = nn.Linear(in_features, out_features, bias=False)
        self.weight_self = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, static_adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor [Batch, Num_Nodes, In_Features]
            static_adj_norm: Normalized static Laplacian [Num_Nodes, Num_Nodes]
        Returns:
            out: Convolved node tensor [Batch, Num_Nodes, Out_Features]
        """
        # Dynamic adaptive transition matrix
        # A_adapt = Softmax(ReLU(E1 * E2^T))
        adapt_adj = F.softmax(F.relu(torch.mm(self.node_embed1, self.node_embed2.t())), dim=-1)

        # 1. Geographic diffusion: A_norm * X
        # Batch matrix multiplication: (B, N, F) -> einsum or bmm
        h_geo = torch.matmul(static_adj_norm.unsqueeze(0), x) # [B, N, F]
        out_geo = self.weight_geo(h_geo)

        # 2. Adaptive corridor diffusion: A_adapt * X
        h_adapt = torch.matmul(adapt_adj.unsqueeze(0), x)
        out_adapt = self.weight_adapt(h_adapt)

        # 3. Self transform
        out_self = self.weight_self(x)

        return F.relu(out_geo + out_adapt + out_self)


class TemporalCausalGRU(nn.Module):
    """
    Causal unidirectional temporal encoder operating on node sequences.
    Ensures zero future leakage.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: [Batch, Num_Nodes, T_steps, Features]
        Returns:
            h_final: [Batch, Num_Nodes, Hidden_Dim] (last causal state)
        """
        b, n, t, f = x_seq.shape
        # Collapse batch and nodes to pass through GRU
        x_flat = x_seq.view(b * n, t, f)
        _, h_n = self.gru(x_flat) # h_n is [1, B*N, Hidden]
        h_final = h_n.squeeze(0).view(b, n, -1)
        return h_final


class CausalConv1d(nn.Module):
    """1D causal convolution with strict left-padding to prevent future data leakage."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch * Nodes, Channels, Time]
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class DilatedCausalWaveNet(nn.Module):
    """
    5-layer Dilated Causal WaveNet encoder covering 32 time steps (8.0 hours).
    Dilations: [1, 2, 4, 8, 16] -> Receptive Field = 1 + 1 + 2 + 4 + 8 + 16 = 32 steps.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.dilations = dilations
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        self.filter_convs = nn.ModuleList([
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.gate_convs = nn.ModuleList([
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.res_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(len(dilations) - 1)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T=32, F] -> flatten to [B*N, F, T]
        b, n, t, f = x.shape
        h = x.contiguous().reshape(b * n, t, f).transpose(1, 2)
        h = self.in_proj(h)

        skip_sum = 0
        for i, (f_conv, g_conv, skip_conv) in enumerate(zip(
            self.filter_convs, self.gate_convs, self.skip_convs
        )):
            residual = h
            f_act = torch.tanh(f_conv(h))
            g_act = torch.sigmoid(g_conv(h))
            gated = f_act * g_act

            skip = skip_conv(gated)
            skip_sum = skip_sum + skip
            if i < len(self.dilations) - 1:
                h = self.res_convs[i](gated) + residual

        # Final causal step at time t
        out = skip_sum[:, :, -1]  # [B*N, hidden_channels]
        return out.reshape(b, n, -1) # [B, N, hidden_channels]


class DirectedAdvectiveGCN(nn.Module):
    """
    Directed Spatio-Temporal Graph Convolution using dual diffusion matrices (P_fwd, P_bwd)
    and an asymmetric learnable adaptive corridor matrix (A_adapt).
    """
    def __init__(self, in_features: int, out_features: int, num_nodes: int = 25, embed_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes

        # Asymmetric node embeddings for directed corridor discovery
        self.e_src = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.e_tgt = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.e_src)
        nn.init.xavier_uniform_(self.e_tgt)

        self.w_fwd = nn.Linear(in_features, out_features, bias=False)
        self.w_bwd = nn.Linear(in_features, out_features, bias=False)
        self.w_adapt = nn.Linear(in_features, out_features, bias=False)
        self.w_self = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, p_fwd: torch.Tensor, p_bwd: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], p_fwd, p_bwd: [N, N] or [B, N, N]
        a_adapt = F.softmax(F.relu(torch.mm(self.e_src, self.e_tgt.t())), dim=-1)

        p_fwd_in = p_fwd.unsqueeze(0) if p_fwd.dim() == 2 else p_fwd
        p_bwd_in = p_bwd.unsqueeze(0) if p_bwd.dim() == 2 else p_bwd

        h_fwd = torch.matmul(p_fwd_in, x)
        h_bwd = torch.matmul(p_bwd_in, x)
        h_adapt = torch.matmul(a_adapt.unsqueeze(0), x)

        out = self.w_fwd(h_fwd) + self.w_bwd(h_bwd) + self.w_adapt(h_adapt) + self.w_self(x)
        return F.leaky_relu(out, negative_slope=0.1)


class AirDefenseInterceptionDecay(nn.Module):
    """
    Physical Air Defense Interception Decay Layer.
    Models physical shoot-down attrition of cruise missiles and UAV swarms
    along flight corridors (P_fwd) by mobile air defense firing groups and SAM batteries:
    P_effective(u, v) = P_fwd(u, v) * sigmoid(W_decay^T [x_u, x_v] + b)
    Dynamically damps phantom alert waves propagating into the deep rear.
    """
    def __init__(self, in_features: int, hidden_dim: int = 16, init_survival_bias: float = 1.75):
        super().__init__()
        self.in_features = in_features
        self.proj_u = nn.Linear(in_features, hidden_dim, bias=False)
        self.proj_v = nn.Linear(in_features, hidden_dim, bias=False)
        self.out_gate = nn.Linear(hidden_dim, 1, bias=True)
        # Default bias initialized to ~1.75 -> sigmoid(1.75) ~ 0.85 (85% baseline survival)
        nn.init.constant_(self.out_gate.bias, init_survival_bias)

    def forward(self, x: torch.Tensor, p_fwd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, F]
            p_fwd: Forward transition matrix [N, N] or [B, N, N]
        Returns:
            p_fwd_eff: Effective decayed corridor matrix [B, N, N]
        """
        b, n, _ = x.shape
        hu = self.proj_u(x) # [B, N, H]
        hv = self.proj_v(x) # [B, N, H]

        # Pairwise broadcast along directed edges
        pair_h = F.relu(hu.unsqueeze(2) + hv.unsqueeze(1)) # [B, N, N, H]
        survival_prob = torch.sigmoid(self.out_gate(pair_h)).squeeze(-1) # [B, N, N] in (0, 1)

        p_fwd_in = p_fwd.unsqueeze(0) if p_fwd.dim() == 2 else p_fwd # [1 or B, N, N]
        p_fwd_eff = p_fwd_in * survival_prob # [B, N, N]
        return p_fwd_eff


class KinematicBearingAttention(nn.Module):
    """
    Kinematic Bearing-Modulated Full-Graph Cross-Attention.
    Replaces 2-hop GCN with full-graph attention modulated by aerodynamic flight corridors:
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d) + B_kinematic ) V
    where B_kinematic(i, j) rewards nodes along forward bearing cone cos(theta_ij) > 0
    and penalizes backward flight corridors.
    Provides instantaneous 1-step receptive field across all Ukrainian regions (e.g. Sumy -> Lviv).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.10,
        beta_fwd: float = 2.0,
        lambda_dist: float = 0.0015,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.beta_fwd = beta_fwd
        self.lambda_dist = lambda_dist
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                                      # [B, N, D]
        coords: Optional[torch.Tensor] = None,                 # [N, 2] (lat, lon)
        threat_vector: Optional[torch.Tensor] = None,          # [B, 2] or [2] (dx, dy)
        external_bias: Optional[torch.Tensor] = None,          # [B, N, N] or [N, N]
    ) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, N, Dh]
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, N, Dh]
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, N, Dh]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, N, N]

        # Compute Kinematic Bearing Bias Matrix if coordinates are available
        if coords is not None:
            # coords: [N, 2] -> diff: c_j - c_i
            # [N, 1, 2] - [1, N, 2] -> vector from i to j
            delta = coords.unsqueeze(0) - coords.unsqueeze(1) # [N, N, 2] (lat_j - lat_i, lon_j - lon_i)
            dist_km = torch.norm(delta, dim=-1) * 111.0       # [N, N] km
            unit_delta = delta / torch.clamp(torch.norm(delta, dim=-1, keepdim=True), min=1e-4) # [N, N, 2]

            # Ingress threat vector: default East-to-West, South-to-Northwest [-0.85, -0.25]
            if threat_vector is None:
                v_threat = torch.tensor([0.0, -1.0], device=x.device, dtype=x.dtype) # Pure East-to-West
            else:
                v_threat = threat_vector.to(device=x.device, dtype=x.dtype)
                if v_threat.dim() == 1:
                    v_threat = v_threat / torch.clamp(torch.norm(v_threat), min=1e-4)

            # Dot product cos(theta_ij) = v_threat . unit(c_j - c_i)
            # When unit vector points in the same direction as threat propagation, cos > 0
            if v_threat.dim() == 1:
                cos_theta = (unit_delta * v_threat.view(1, 1, 2)).sum(dim=-1) # [N, N]
            else:
                cos_theta = (unit_delta.unsqueeze(0) * v_threat.view(b, 1, 1, 2)).sum(dim=-1) # [B, N, N]

            # Bearing bias: forward cone boost, distance transit delay, and backward damping
            fwd_cone = F.relu(cos_theta)
            bwd_mask = (cos_theta < -0.35).float()
            b_kin = self.beta_fwd * fwd_cone - self.lambda_dist * dist_km - 6.0 * bwd_mask

            if b_kin.dim() == 2:
                b_kin = b_kin.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
            elif b_kin.dim() == 3:
                b_kin = b_kin.unsqueeze(1) # [B, 1, N, N]

            scores = scores + b_kin

        if external_bias is not None:
            ext_b = external_bias.unsqueeze(1) if external_bias.dim() == 3 else external_bias.unsqueeze(0).unsqueeze(0)
            scores = scores + ext_b

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v) # [B, H, N, Dh]
        context = context.transpose(1, 2).contiguous().reshape(b, n, d) # [B, N, D]
        return self.out_proj(context)


class KinematicTrajectoryTransformer(nn.Module):
    """
    Multi-layer Kinematic Trajectory Transformer Block.
    Combines KinematicBearingAttention with pre-LayerNorm, FeedForward projection,
    and residual skip connections.
    """
    def __init__(self, d_model: int, num_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = KinematicBearingAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        threat_vector: Optional[torch.Tensor] = None,
        external_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        h_norm = self.norm1(x)
        attn_out = self.attn(h_norm, coords=coords, threat_vector=threat_vector, external_bias=external_bias)
        x = x + attn_out

        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ContinuousWeibullSurvivalHead(nn.Module):
    """
    Continuous Weibull Survival and Hazard Rate Prediction Head.
    Outputs continuous scale (eta > 0) and shape (kappa > 0) parameters:
    - Time-to-Impact: E[T] = eta * Gamma(1 + 1/kappa)
    - Survival Function: S(t) = exp( - (t / eta)^kappa )
    - Hazard Rate: lambda(t) = (kappa / eta) * (t / eta)^(kappa - 1)
    Permanently resolves false-alarm window boundary penalties.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.15),
            nn.Linear(hidden_dim, 2), # [log_eta, log_kappa]
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_tti: Expected continuous hours until impact in [0.0, 6.0h]
            eta: Scale parameter [B, N]
            kappa: Shape parameter [B, N]
        """
        params = self.fc(h)
        # eta in [0.2, 8.0 hours]
        eta = F.softplus(params[..., 0]) + 0.20
        # kappa in [0.6, 3.5]
        kappa = F.softplus(params[..., 1]) + 0.60

        # Mean of Weibull: eta * Gamma(1 + 1/kappa)
        # Using Ramanujan's approximation for Gamma(1 + x)
        inv_k = 1.0 / kappa
        # Safe approximation: Gamma(1 + z) ~ 1.0 - 0.5772 * z + 0.989 * z^2
        gamma_approx = 1.0 - 0.5772 * inv_k + 0.989 * torch.pow(inv_k, 2)
        mean_tti = eta * gamma_approx.clamp(0.85, 1.25)
        pred_tti = torch.clamp(mean_tti, 0.0, 6.0)

        return pred_tti, eta, kappa


def pcgrad_backward(
    loss_tactical: torch.Tensor,
    loss_continuous: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    accum_steps: int = 1,
) -> float:
    """
    Projected Conflicting Gradients (PCGrad) for Multi-Task Optimization:
    Projects tactical onset focal gradients and continuous duration/hazard gradients
    onto each other's orthogonal planes if they conflict (<g_1, g_2> < 0).
    Eliminates multi-task gradient cancellation.
    Supports gradient accumulation across multiple mini-batches.

    Returns:
        total_loss_val: float scalar value of the combined loss
    """
    total_loss_val = (loss_tactical + loss_continuous).item()

    # Save existing accumulated gradients if any
    existing_grads = []
    if accum_steps > 1:
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    existing_grads.append(p.grad.detach().clone())
                else:
                    existing_grads.append(None)

    # 1. Compute tactical gradients
    optimizer.zero_grad()
    loss_tactical.backward(retain_graph=True)
    g_tactical = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                g_tactical.append(p.grad.detach().clone().flatten())
            else:
                g_tactical.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    g1 = torch.cat(g_tactical)

    # 2. Compute continuous gradients
    optimizer.zero_grad()
    loss_continuous.backward()
    g_continuous = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                g_continuous.append(p.grad.detach().clone().flatten())
            else:
                g_continuous.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    g2 = torch.cat(g_continuous)

    # 3. Conflict projection
    dot = torch.dot(g1, g2)
    if dot < 0:
        norm_sq_g2 = torch.dot(g2, g2) + 1e-8
        norm_sq_g1 = torch.dot(g1, g1) + 1e-8
        g1_proj = g1 - (dot / norm_sq_g2) * g2
        g2_proj = g2 - (dot / norm_sq_g1) * g1
    else:
        g1_proj = g1
        g2_proj = g2

    # 4. Scale by accum_steps, restore existing accumulated gradients, and assign
    g_final = (g1_proj + g2_proj) / max(1, accum_steps)
    idx = 0
    p_idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            g_chunk = g_final[idx : idx + numel].view_as(p)
            if accum_steps > 1 and existing_grads[p_idx] is not None:
                p.grad = (existing_grads[p_idx] + g_chunk).clone()
            else:
                p.grad = g_chunk.clone()
            idx += numel
            p_idx += 1

    return total_loss_val

