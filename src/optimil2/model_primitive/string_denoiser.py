"""The string denoiser: a small bidirectional transformer over character positions.

The string twin of denoiser.py. Input is a noisy row of character tokens and
the noise level t in [0, 1]. Output is, for every position, logits over the
classes of the CLEAN string. All positions are predicted at once from all
positions; nothing is generated left to right.

The network keeps one vector per position. Each layer does two things:

1. Positions attend to all other positions (plain multi-head attention).
2. A feed-forward block on every position vector.

This is the graph transformer's layer with the pair vectors taken out: the
same pre-norm blocks, the same way of feeding in the noise level, and the
same default width, depth and number of heads. A string has no pair state,
so the two things that differ are what the comparison is about: there is no
pair stream, and there are position embeddings.

Decisions:
- Bidirectional: no causal mask, so every position sees the whole noisy
  string, as every atom slot sees the whole noisy graph.
- Learned position embeddings, one vector per position, added to the token
  embedding. Unlike the graph, a string is nothing but its order, so the
  network must know where each token is. Learned absolute positions are the
  simplest way, and enough for a fixed length of 128. Relative schemes such
  as RoPE might suit SMILES better, since the anion starts at a different
  position in every IL; that is something to try, not to start with.
- PAD is an ordinary token. There is no padding mask anywhere: the network
  has to predict PAD for unused positions, which is how it chooses the
  length of the string. The graph model treats EMPTY slots the same way.
- Default sizes give 1.27M parameters, against 1.32M for the graph
  transformer. The difference is the graph's pair stream (0.07M) less the
  position embeddings here (0.02M). It is 4% and no attempt is made to
  close it; doing so would mean giving the two networks different widths.
- Cost is not matched, only parameters. Attention here runs over 128
  positions and the graph's over 56 slots plus a state for each of its
  3,136 pairs.
"""

import torch
import torch.nn.functional as F
from torch import nn

from optimil2.model_primitive.denoiser import feed_forward, timestep_embedding


class Layer(nn.Module):
    """One round of attention between positions and a feed-forward block."""

    def __init__(self, hidden: int, hidden_t: int, num_heads: int, ff_multiplier: int):
        super().__init__()
        if hidden % num_heads:
            raise ValueError(f"hidden ({hidden}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.time = nn.Linear(hidden_t, hidden)
        self.norm_attention = nn.LayerNorm(hidden)
        self.query_key_value = nn.Linear(hidden, 3 * hidden)
        self.attention_out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.ff = feed_forward(hidden, ff_multiplier)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """x is (B, L, hidden), time is (B, hidden_t)."""
        batch, length, hidden = x.shape
        h = self.norm_attention(x + self.time(time).unsqueeze(1))
        # (3, B, heads, L, hidden / heads): queries, keys and values of every head
        query, key, value = self.query_key_value(h).view(batch, length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value)  # no mask: every position sees every position
        x = x + self.attention_out(attended.transpose(1, 2).reshape(batch, length, hidden))
        return x + self.ff(self.norm(x))


class StringTransformer(nn.Module):
    """Predicts the clean string from a noisy one."""

    def __init__(
        self,
        num_classes: int,
        length: int,
        hidden: int = 128,
        hidden_t: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_multiplier: int = 2,
    ):
        super().__init__()
        if hidden_t % 2:
            raise ValueError(f"hidden_t ({hidden_t}) must be even")
        self.hidden_t = hidden_t
        self.embed = nn.Embedding(num_classes, hidden)
        self.positions = nn.Parameter(torch.randn(length, hidden))  # same initial scale as the token embedding
        self.embed_t = nn.Sequential(nn.Linear(hidden_t, hidden_t), nn.GELU(), nn.Linear(hidden_t, hidden_t))
        self.layers = nn.ModuleList(Layer(hidden, hidden_t, num_heads, ff_multiplier) for _ in range(num_layers))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return logits over clean classes.

        tokens: (B, L) character class indices of the noisy string, with L
            the length this network was built for.
        t: (B,) noise level in [0, 1]; 0 is clean, 1 is pure noise.

        Returns (B, L, num_classes).
        """
        time = self.embed_t(timestep_embedding(t, self.hidden_t))
        h = self.embed(tokens) + self.positions
        for layer in self.layers:
            h = layer(h, time)
        return self.out(h)
