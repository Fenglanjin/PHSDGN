"""PHSDGN model definition."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import PhysicsDecoderLayer, PhysicsEncoderLayer, TemporalAttentionReadout


class PHSDGN(nn.Module):
    """Physics-embedded high-order spatiotemporal dynamic graph network."""

    def __init__(
        self,
        F: int,
        num_nodes: int,
        tem_size: int,
        static_adj: torch.Tensor,
        K: int = 3,
        Kt: int = 7,
        hidden_dim: int = 64,
        n_layers: int = 2,
        alpha: float = 0.1,
        use_dynamic_graph: bool = True,
        use_physics_conv: bool = True,
        use_std_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_dynamic_graph = use_dynamic_graph
        self.use_physics_conv = use_physics_conv
        self.use_std_pooling = use_std_pooling

        self.register_buffer("static_adj", static_adj)

        self.input_proj = nn.Sequential(
            nn.Conv2d(F, hidden_dim, kernel_size=(1, 1), bias=True),
            nn.LeakyReLU(),
        )

        self.enc_layers = nn.ModuleList([
            PhysicsEncoderLayer(
                num_nodes,
                tem_size,
                K,
                Kt,
                hidden_dim,
                use_dynamic_graph=use_dynamic_graph,
                use_physics_conv=use_physics_conv,
                alpha=alpha,
            )
            for _ in range(max(1, n_layers - 1))
        ])

        self.dec_layer = PhysicsDecoderLayer(
            num_nodes,
            tem_size,
            K,
            Kt,
            hidden_dim,
            use_dynamic_graph=use_dynamic_graph,
            use_physics_conv=use_physics_conv,
            alpha=alpha,
        )

        self.temp_readout = TemporalAttentionReadout(hidden_dim, tem_size, use_std_pooling=use_std_pooling)
        flatten_dim = num_nodes * hidden_dim if not use_std_pooling else num_nodes * 2 * hidden_dim
        self.regressor = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)

        dyn_adj = None
        for enc in self.enc_layers:
            z, dyn_adj = enc(z, self.static_adj)

        z, dyn_adj, _ = self.dec_layer(z, self.static_adj)
        z_temp = self.temp_readout(z)
        z_flat = z_temp.view(z_temp.size(0), -1)
        h_hat = self.regressor(z_flat).squeeze(-1)

        return h_hat, dyn_adj


# Backward-compatible alias for checkpoints and older scripts.
PIHST_GNet = PHSDGN
