"""The denoiser: a small dense graph transformer.

Input is a noisy graph (atom class per slot, bond class per slot pair) and the
noise level t in [0, 1]. Output is, for every slot and every pair, logits
over the classes of the CLEAN graph.

The network keeps one vector per slot (width hidden_x) and one per slot pair
(width hidden_e). Each layer does three things:

1. Slots attend to all other slots. The attention score of slot i for slot j
   is the usual query-key product plus a bias computed from the pair vector
   (i, j), which is how bonds steer the attention.
2. Each pair vector is updated from its two slots and from the attention
   scores between them.
3. A feed-forward block on every slot vector and every pair vector.

Decisions:
- Dense, not sparse message passing: bonds are part of what is generated, so
  every pair needs a state, bonded or not. With 56 slots this is 3,136 pairs
  per graph, which is why hidden_e is much smaller than hidden_x.
- Simpler than the DiGress network. DiGress multiplies queries and keys per
  feature and modulates them with the pair features, which keeps an
  (N, N, hidden_x) tensor per layer; here attention is one score per head, an
  (N, N, heads) tensor, about 16 times less memory at the default sizes.
- EMPTY slots are ordinary tokens. There is no padding mask anywhere: the
  network has to predict EMPTY for unused slots, which is how it chooses the
  number of atoms.
- No positional encoding, so the output does not depend on slot order
  (permuting the slots permutes the output the same way). The cost is that
  the network cannot tell symmetric atoms apart except through the noise.
- Left out, to keep this minimal: the extra input features DiGress computes
  from the noisy graph (ring counts, spectral features, valency). A network
  of this kind cannot count rings on its own, so these are the first thing
  to add if samples have too many broken rings.
- The pair logits are averaged with their transpose at the end, so the
  prediction for (i, j) equals the one for (j, i).
- Default sizes give about 1.3M parameters.
"""

import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of t, shape (B,) with values in [0, 1], as (B, dim)."""
    frequencies = torch.exp(-math.log(10_000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = 1000 * t.float().unsqueeze(1) * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def feed_forward(dim: int, multiplier: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, multiplier * dim), nn.GELU(), nn.Linear(multiplier * dim, dim))


class Layer(nn.Module):
    """One round of slot attention, pair update and feed-forward blocks."""

    def __init__(self, hidden_x: int, hidden_e: int, hidden_t: int, num_heads: int, ff_multiplier: int):
        super().__init__()
        if hidden_x % num_heads:
            raise ValueError(f"hidden_x ({hidden_x}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.time = nn.Linear(hidden_t, hidden_x)
        self.norm_attention = nn.LayerNorm(hidden_x)
        self.query_key_value = nn.Linear(hidden_x, 3 * hidden_x)
        self.pair_bias = nn.Linear(hidden_e, num_heads)
        self.attention_out = nn.Linear(hidden_x, hidden_x)
        self.norm_x = nn.LayerNorm(hidden_x)
        self.ff_x = feed_forward(hidden_x, ff_multiplier)

        self.slot_to_pair = nn.Linear(hidden_x, hidden_e)
        self.score_to_pair = nn.Linear(num_heads, hidden_e)
        self.norm_e = nn.LayerNorm(hidden_e)
        self.ff_e = feed_forward(hidden_e, ff_multiplier)

    def forward(self, x: torch.Tensor, e: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x is (B, N, hidden_x), e is (B, N, N, hidden_e), time is (B, hidden_t)."""
        batch, slots, hidden_x = x.shape
        heads = self.num_heads

        # 1. Slots attend to slots, steered by the pair vectors.
        h = self.norm_attention(x + self.time(time).unsqueeze(1))
        query, key, value = self.query_key_value(h).view(batch, slots, 3, heads, hidden_x // heads).unbind(dim=2)
        scores = torch.einsum("bihd,bjhd->bijh", query, key) / math.sqrt(hidden_x // heads)
        scores = scores + self.pair_bias(e)  # (B, N, N, heads): score of slot i for slot j
        attended = torch.einsum("bijh,bjhd->bihd", scores.softmax(dim=2), value)
        x = x + self.attention_out(attended.reshape(batch, slots, hidden_x))
        x = x + self.ff_x(self.norm_x(x))

        # 2. Pairs read their two slots and the scores between them.
        from_slots = self.slot_to_pair(x)
        e = e + from_slots.unsqueeze(1) + from_slots.unsqueeze(2) + self.score_to_pair(scores)
        e = e + self.ff_e(self.norm_e(e))
        return x, e


class GraphTransformer(nn.Module):
    """Predicts the clean graph from a noisy one."""

    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        hidden_x: int = 128,
        hidden_e: int = 32,
        hidden_t: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_multiplier: int = 2,
    ):
        super().__init__()
        if hidden_t % 2:
            raise ValueError(f"hidden_t ({hidden_t}) must be even")
        self.hidden_t = hidden_t
        self.embed_x = nn.Embedding(num_atom_classes, hidden_x)
        self.embed_e = nn.Embedding(num_bond_classes, hidden_e)
        self.embed_t = nn.Sequential(nn.Linear(hidden_t, hidden_t), nn.GELU(), nn.Linear(hidden_t, hidden_t))
        self.layers = nn.ModuleList(
            Layer(hidden_x, hidden_e, hidden_t, num_heads, ff_multiplier) for _ in range(num_layers)
        )
        self.out_x = nn.Sequential(nn.LayerNorm(hidden_x), nn.Linear(hidden_x, hidden_x), nn.GELU(), nn.Linear(hidden_x, num_atom_classes))
        self.out_e = nn.Sequential(nn.LayerNorm(hidden_e), nn.Linear(hidden_e, hidden_e), nn.GELU(), nn.Linear(hidden_e, num_bond_classes))

    def forward(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits over clean classes.

        x: (B, N) atom class indices of the noisy graph.
        e: (B, N, N) bond class indices of the noisy graph, symmetric.
        t: (B,) noise level in [0, 1]; 0 is clean, 1 is pure noise.

        Returns (B, N, num_atom_classes) and (B, N, N, num_bond_classes).
        The pair logits are symmetric in the two slots. Their diagonal
        (a slot paired with itself) is meaningless and should be ignored.
        """
        time = self.embed_t(timestep_embedding(t, self.hidden_t))
        h_x, h_e = self.embed_x(x), self.embed_e(e)
        for layer in self.layers:
            h_x, h_e = layer(h_x, h_e, time)
        logits_e = self.out_e(h_e)
        return self.out_x(h_x), (logits_e + logits_e.transpose(1, 2)) / 2
