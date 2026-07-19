"""
Single-file Budiarjo actor-critic model.

Model semantics:
    input features: (batch_size, num_hosts, input_dim)
    actor output:   (batch_size, num_hosts, 2)
    critic output:  scalar value
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


CPU_DEVICE = torch.device("cpu")


class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.module(inputs)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads

        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1.0 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(
            torch.empty(n_heads, input_dim, key_dim)
        )
        self.W_key = nn.Parameter(
            torch.empty(n_heads, input_dim, key_dim)
        )
        self.W_val = nn.Parameter(
            torch.empty(n_heads, input_dim, val_dim)
        )
        self.W_out = nn.Parameter(
            torch.empty(n_heads, val_dim, embed_dim)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        for parameter in self.parameters():
            stdv = 1.0 / math.sqrt(parameter.size(-1))
            nn.init.uniform_(parameter, -stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shape_h = (
            self.n_heads,
            batch_size,
            graph_size,
            -1,
        )
        shape_q = (
            self.n_heads,
            batch_size,
            n_query,
            -1,
        )

        queries = torch.matmul(
            qflat,
            self.W_query,
        ).view(shape_q)

        keys = torch.matmul(
            hflat,
            self.W_key,
        ).view(shape_h)

        values = torch.matmul(
            hflat,
            self.W_val,
        ).view(shape_h)

        compatibility = self.norm_factor * torch.matmul(
            queries,
            keys.transpose(2, 3),
        )

        # Preserves the original implementation:
        # mask is accepted but not applied.
        attention = torch.softmax(
            compatibility,
            dim=-1,
        )

        heads = torch.matmul(
            attention,
            values,
        )

        output = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                -1,
                self.n_heads * self.val_dim,
            ),
            self.W_out.view(
                -1,
                self.embed_dim,
            ),
        ).view(
            batch_size,
            n_query,
            self.embed_dim,
        )

        return output


class Normalization(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.normalizer = nn.InstanceNorm1d(
            embed_dim,
            affine=True,
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        for parameter in self.parameters():
            stdv = 1.0 / math.sqrt(parameter.size(-1))
            nn.init.uniform_(parameter, -stdv, stdv)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.normalizer(
            inputs.permute(0, 2, 1)
        ).permute(0, 2, 1)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
    ) -> None:
        super().__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                )
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        embed_dim,
                        feed_forward_hidden,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        feed_forward_hidden,
                        embed_dim,
                    ),
                )
                if feed_forward_hidden > 0
                else nn.Linear(
                    embed_dim,
                    embed_dim,
                )
            ),
            Normalization(embed_dim),
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: Optional[int] = None,
        feed_forward_hidden: int = 512,
    ) -> None:
        super().__init__()

        self.init_embed = (
            nn.Linear(node_dim, embed_dim)
            if node_dim is not None
            else None
        )

        self.layers = nn.Sequential(
            *[
                MultiHeadAttentionLayer(
                    n_heads=n_heads,
                    embed_dim=embed_dim,
                    feed_forward_hidden=feed_forward_hidden,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = (
            self.init_embed(x)
            if self.init_embed is not None
            else x
        )

        embeddings = self.layers(hidden)

        graph_embedding = embeddings.mean(dim=1)

        return embeddings, graph_embedding


class Agent(nn.Module):
    """Original Budiarjo per-host binary actor."""

    def __init__(
        self,
        n_heads: int = 8,
        n_gae_layers: int = 3,
        input_dim: int = 11,
        embed_dim: int = 128,
        gae_ff_hidden: int = 512,
        tanh_clip: float = 10,
        device: torch.device | str = CPU_DEVICE,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.key_size = embed_dim // n_heads
        self.val_size = embed_dim // n_heads

        self.gae = GraphAttentionEncoder(
            n_heads=n_heads,
            n_layers=n_gae_layers,
            embed_dim=embed_dim,
            node_dim=input_dim,
            feed_forward_hidden=gae_ff_hidden,
        )

        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # Initialize the module on the requested device.
        # Do not store self.device because model.to(...) may change it later.
        self.to(torch.device(device))

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        features = features.to(
            device=device,
            dtype=torch.float32,
        )

        if mask is not None:
            mask = mask.to(device=device)

        embeddings, _ = self.gae(features)

        logits = self.prob_head(embeddings)

        probabilities = torch.softmax(
            logits,
            dim=2,
        )

        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities.clamp_min(1e-12)
            )
        )

        return probabilities, entropy


class Critic(nn.Module):
    """Original Budiarjo graph-level critic."""

    def __init__(
        self,
        n_heads: int = 8,
        n_gae_layers: int = 3,
        input_dim: int = 11,
        embed_dim: int = 128,
        gae_ff_hidden: int = 512,
        tanh_clip: float = 10,
        device: torch.device | str = CPU_DEVICE,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.key_size = embed_dim // n_heads
        self.val_size = embed_dim // n_heads

        self.gae = GraphAttentionEncoder(
            n_heads=n_heads,
            n_layers=n_gae_layers,
            embed_dim=embed_dim,
            node_dim=input_dim,
            feed_forward_hidden=gae_ff_hidden,
        )

        self.value_layers = nn.Sequential(
            nn.Linear(embed_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        # Initialize on the requested device without storing self.device.
        self.to(torch.device(device))

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        features = features.to(
            device=device,
            dtype=torch.float32,
        )

        _, environment_embeddings = self.gae(features)

        values = self.value_layers(
            environment_embeddings
        ).sum()

        return -values


class Budiarjo(nn.Module):
    """
    Single model wrapper around the original Budiarjo Agent and Critic.

    Input:
        features:
            (batch_size, num_hosts, input_dim)

            A two-dimensional input with shape
            (num_hosts, input_dim) is automatically converted to a
            batch of size one.

        mask:
            Optional tensor accepted for compatibility with the original
            actor interface.

    Output:
        probabilities:
            (batch_size, num_hosts, 2)

        value:
            Scalar tensor, preserving the original Critic behavior.
    """

    def __init__(
        self,
        n_heads: int = 8,
        n_gae_layers: int = 3,
        input_dim: int = 11,
        embed_dim: int = 128,
        gae_ff_hidden: int = 512,
        tanh_clip: float = 10,
        device: torch.device | str = CPU_DEVICE,
    ) -> None:
        super().__init__()

        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.gae_ff_hidden = gae_ff_hidden
        self.tanh_clip = tanh_clip

        # Build the submodules on CPU first.
        # The complete wrapper is moved once at the end.
        self.agent = Agent(
            n_heads=n_heads,
            n_gae_layers=n_gae_layers,
            input_dim=input_dim,
            embed_dim=embed_dim,
            gae_ff_hidden=gae_ff_hidden,
            tanh_clip=tanh_clip,
            device=CPU_DEVICE,
        )

        self.critic = Critic(
            n_heads=n_heads,
            n_gae_layers=n_gae_layers,
            input_dim=input_dim,
            embed_dim=embed_dim,
            gae_ff_hidden=gae_ff_hidden,
            tanh_clip=tanh_clip,
            device=CPU_DEVICE,
        )

        self.to(torch.device(device))

    def _prepare_features(
        self,
        features,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=device,
        )

        if features.ndim == 2:
            features = features.unsqueeze(0)

        if features.ndim != 3:
            raise ValueError(
                "features must have shape "
                "(batch_size, num_hosts, input_dim) "
                "or (num_hosts, input_dim); "
                f"got {tuple(features.shape)}"
            )

        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got {features.size(-1)}"
            )

        return features
    def _prepare_mask(
        self,
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        device = next(self.parameters()).device

        return mask.to(device=device)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._prepare_features(features)
        mask = self._prepare_mask(mask)

        probabilities, _ = self.agent(
            features,
            mask,
        )

        value = self.critic(features)

        return probabilities, value

    def forward_with_entropy(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        features = self._prepare_features(features)
        mask = self._prepare_mask(mask)

        probabilities, entropy = self.agent(
            features,
            mask,
        )

        value = self.critic(features)

        return probabilities, entropy, value

    # Runner lifecycle compatibility methods.
    def reset_episode(self) -> None:
        pass

    def start_rollout(self) -> None:
        pass

    def reset_evaluation_pointer(self) -> None:
        pass

    def forward2(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            features,
            mask,
        )

    def peek(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            features,
            mask,
        )