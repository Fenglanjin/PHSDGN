"""Core neural operators used by PHSDGN.

Tensor convention: inputs and hidden states use [batch, channels, num_nodes, time].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Conv2d, Parameter


class PhysicsConv(nn.Module):
    """Physics-embedded high-order spatiotemporal propagation operator.

    The operator combines three branches:
    1. graph-Laplacian Chebyshev propagation for multi-hop spatial diffusion;
    2. temporal differencing followed by one-dimensional convolution;
    3. an identity/compensation branch.
    """

    def __init__(self, c_in: int, c_out: int, K: int, Kt: int, use_physics_conv: bool = True) -> None:
        super().__init__()
        self.K = K
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.use_physics_conv = use_physics_conv

        if use_physics_conv:
            self.spatial_conv = Conv2d((K + 1) * c_in, c_out, kernel_size=(1, 1), bias=True)
            self.temp_diff_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, Kt // 2), bias=True)
            self.identity_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, Kt // 2), bias=True)
        else:
            self.bypass = Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    @staticmethod
    def _normalize_laplacian(adj: torch.Tensor) -> torch.Tensor:
        """Return a scaled graph Laplacian used by Chebyshev propagation."""
        deg = adj.sum(dim=-1)
        laplacian = torch.diag_embed(deg) - adj
        identity = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        return laplacian - identity

    @staticmethod
    def _chebyshev_bases(x: torch.Tensor, laplacian: torch.Tensor, K: int) -> torch.Tensor:
        """Build Chebyshev bases T_0...T_K over the node dimension."""
        t0 = x
        bases = [t0]
        if K >= 1:
            t1 = torch.einsum("bnm,bcmt->bcnt", laplacian, x)
            bases.append(t1)
            for _ in range(2, K + 1):
                t2 = 2.0 * torch.einsum("bnm,bcmt->bcnt", laplacian, t1) - t0
                bases.append(t2)
                t0, t1 = t1, t2
        return torch.cat(bases, dim=1)

    @staticmethod
    def _first_order_diff(x: torch.Tensor) -> torch.Tensor:
        """Compute first-order temporal differences while preserving sequence length."""
        diff = x[:, :, :, 1:] - x[:, :, :, :-1]
        return F.pad(diff, (1, 0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply physics-embedded propagation.

        Args:
            x: Hidden tensor with shape [B, C, N, T].
            adj: Dynamic or static adjacency tensor with shape [B, N, N].
        """
        if not self.use_physics_conv:
            return self.bypass(x)

        laplacian = self._normalize_laplacian(adj)
        cheb = self._chebyshev_bases(x, laplacian, self.K)
        out_spatial = self.spatial_conv(cheb)

        diff = self._first_order_diff(x)
        out_temporal = self.temp_diff_conv(diff)
        out_identity = self.identity_conv(x)

        return out_identity + out_spatial + out_temporal


class SpatialAttention(nn.Module):
    """Generate state-dependent graph weights from hidden PMU features."""

    def __init__(self, num_nodes: int, tem_size: int, channels: int) -> None:
        super().__init__()
        self.channel_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.conv1 = Conv2d(1, 1, kernel_size=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1), bias=False)
        self.w = Parameter(torch.rand(tem_size, 1))
        self.b = Parameter(torch.zeros(num_nodes, num_nodes))
        self.v = Parameter(torch.rand(num_nodes, num_nodes))
        self.bn = BatchNorm1d(num_nodes)
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        f1 = self.conv1(x).squeeze(1)
        f2 = self.conv2(x.permute(0, 3, 1, 2)).squeeze(1)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = self.bn(logits.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.softmax(logits, dim=-1)


class PhysicsEncoderLayer(nn.Module):
    """Encoder block with topology-masked dynamic graph propagation."""

    def __init__(
        self,
        num_nodes: int,
        tem_size: int,
        K: int,
        Kt: int,
        channels: int,
        use_dynamic_graph: bool = True,
        use_physics_conv: bool = True,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_dynamic_graph = use_dynamic_graph
        self.alpha = alpha
        if use_dynamic_graph:
            self.satt = SpatialAttention(num_nodes, tem_size, channels)
        self.pconv = PhysicsConv(channels, channels, K=K, Kt=Kt, use_physics_conv=use_physics_conv)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static = static_adj / (static_adj.sum(dim=-1, keepdim=True) + 1e-8)
        if self.use_dynamic_graph:
            dyn_adj = self.satt(x)
            adj = self.alpha * static + (1.0 - self.alpha) * (dyn_adj * static)
        else:
            batch_size = x.size(0)
            dyn_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
            adj = dyn_adj

        out = F.leaky_relu(self.pconv(x, adj))
        return out, dyn_adj


class PhysicsDecoderLayer(nn.Module):
    """Decoder block that applies the final topology-masked propagation step."""

    def __init__(
        self,
        num_nodes: int,
        tem_size: int,
        K: int,
        Kt: int,
        channels: int,
        use_dynamic_graph: bool = True,
        use_physics_conv: bool = True,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_dynamic_graph = use_dynamic_graph
        self.alpha = alpha
        if use_dynamic_graph:
            self.satt = SpatialAttention(num_nodes, tem_size, channels)
        self.pconv = PhysicsConv(channels, channels, K=K, Kt=Kt, use_physics_conv=use_physics_conv)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        batch_size = x.size(0)
        static = static_adj / (static_adj.sum(dim=-1, keepdim=True) + 1e-8)
        if self.use_dynamic_graph:
            dyn_adj = self.satt(x)
            adj = self.alpha * static + (1.0 - self.alpha) * (dyn_adj * static)
        else:
            dyn_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
            adj = dyn_adj

        x_sp = F.leaky_relu(self.pconv(x, adj))
        out = self.bn(x_sp)
        return out, dyn_adj, None


class TemporalAttentionReadout(nn.Module):
    """Temporal attention readout with optional fluctuation statistics."""

    def __init__(self, channels: int, tem_size: int, use_std_pooling: bool = False) -> None:
        super().__init__()
        self.use_std_pooling = use_std_pooling
        self.q = Parameter(torch.randn(tem_size))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        _, _, _, num_steps = Z.shape
        z_mean = Z.mean(dim=2)
        score = z_mean * self.q.view(1, 1, num_steps)
        beta = F.softmax(score, dim=-1)
        z_temp = (Z * beta.unsqueeze(2)).sum(dim=-1)

        if self.use_std_pooling:
            z_temp_std = Z.std(dim=-1)
            z_temp = torch.cat([z_temp, z_temp_std], dim=1)

        return z_temp
