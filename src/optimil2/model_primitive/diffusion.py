"""Discrete graph diffusion: add noise, compute the training loss, sample.

`GraphDiffusion` wraps a denoiser with the noise process of noise.py. A graph
is the pair (x, e) described in data/processing/molgraph.py: x is (B, N) atom
class indices and e is (B, N, N) bond class indices, symmetric with a zero
diagonal.

- `add_noise` corrupts clean graphs to a chosen step t.
- `forward` runs the denoiser on noisy graphs: logits over the clean classes.
- `loss` is what training would minimise: cross-entropy between those logits
  and the clean graph, at a random step per graph.
- `sample` starts from pure noise and walks back to step 0, each time drawing
  x_{t-1} from the posterior under the denoiser's prediction.

`build_model` creates one with random weights, sized for the processed data.
There is no training loop.

Every method that draws random numbers takes an optional torch.Generator for
reproducibility; it must live on the same device as the model.

The noise process itself (schedule, transitions, posterior, T) is in noise.py
and is shared with the string model in string_diffusion.py. What is specific
to graphs is here.

Decisions:
- Bonds are noised and sampled on the upper triangle and mirrored, so every
  noisy graph is a valid undirected graph. The diagonal is always NO_BOND.
- EMPTY and NO_BOND are diffused like any other class, over all N slots and
  all pairs. Nothing is masked, so a sample's number of atoms is whatever
  number of slots ends up non-empty.
- The loss averages over all slots and over all pairs, and weights the bond
  term by 5 as DiGress does. Most of these terms are easy (EMPTY, NO_BOND);
  whether that drowns the informative ones is a question for training.
- The loss is the plain cross-entropy, not the variational bound. DiGress
  trains the same way and uses the bound only to report likelihoods.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from optimil2.data.graph_dataset import train_marginals
from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR
from optimil2.data.processing.molgraph import Vocab
from optimil2.model_primitive.denoiser import GraphTransformer
from optimil2.model_primitive.noise import (
    NUM_STEPS,
    cosine_alpha_bar,
    noise_distribution,
    noisy_probs,
    previous_probs,
    sample_categorical,
    symmetrize,
)


@dataclass(frozen=True)
class ModelConfig:
    num_steps: int = NUM_STEPS  # diffusion steps T
    edge_loss_weight: float = 5.0  # weight of the bond term in the loss
    hidden_x: int = 128  # width of the per-slot vectors
    hidden_e: int = 32  # width of the per-pair vectors
    hidden_t: int = 128  # width of the noise-level embedding
    num_layers: int = 8
    num_heads: int = 8
    ff_multiplier: int = 2  # feed-forward blocks widen by this factor


class GraphDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        atom_marginals: torch.Tensor,
        bond_marginals: torch.Tensor,
        num_slots: int,
        num_steps: int = NUM_STEPS,
        edge_loss_weight: float = 5.0,
    ):
        """denoiser maps (x, e, t in [0, 1]) to logits; the marginals are the noise distributions.

        num_slots is the number of atom slots N of the graphs the marginals
        were measured on. Samples have this size: the share of EMPTY in the
        atom marginals only means the right number of atoms for this N.
        """
        super().__init__()
        self.denoiser = denoiser
        self.num_slots = num_slots
        self.num_steps = num_steps
        self.edge_loss_weight = edge_loss_weight
        self.register_buffer("atom_marginals", noise_distribution(atom_marginals, "atom"))
        self.register_buffer("bond_marginals", noise_distribution(bond_marginals, "bond"))
        self.register_buffer("alpha_bar", cosine_alpha_bar(num_steps).float())

    def add_noise(
        self, x0: torch.Tensor, e0: torch.Tensor, t: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw (x_t, e_t) from q(. | x0, e0). t is (B,) integer steps in 0..T; step 0 returns the input."""
        keep = self.alpha_bar[t]
        x_probs = noisy_probs(x0, keep, self.atom_marginals)
        e_probs = noisy_probs(e0, keep, self.bond_marginals)
        return sample_categorical(x_probs, generator), symmetrize(sample_categorical(e_probs, generator))

    def forward(self, x_t: torch.Tensor, e_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits over the clean classes for noisy graphs at integer steps t, shape (B,).

        Returns (B, N, atom classes) and (B, N, N, bond classes).
        """
        return self.denoiser(x_t, e_t, t / self.num_steps)

    def loss(self, x0: torch.Tensor, e0: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Cross-entropy of the denoiser's prediction of (x0, e0) at one random step per graph.

        Returns "loss" (to minimise) and its two parts, "node" and "edge".
        """
        t = torch.randint(1, self.num_steps + 1, (len(x0),), generator=generator, device=x0.device)
        x_t, e_t = self.add_noise(x0, e0, t, generator)
        x_logits, e_logits = self(x_t, e_t, t)
        rows, cols = torch.triu_indices(e0.shape[1], e0.shape[1], offset=1, device=e0.device)
        node = F.cross_entropy(x_logits.flatten(0, 1), x0.flatten())
        edge = F.cross_entropy(e_logits[:, rows, cols].flatten(0, 1), e0[:, rows, cols].flatten())
        return {"loss": node + self.edge_loss_weight * edge, "node": node, "edge": edge}

    def sample_noise(self, num_graphs: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw graphs of pure noise: every slot and pair independently from the marginals."""
        slots = self.num_slots
        x = sample_categorical(self.atom_marginals.expand(num_graphs, slots, -1), generator)
        e = sample_categorical(self.bond_marginals.expand(num_graphs, slots, slots, -1), generator)
        return x, symmetrize(e)

    @torch.no_grad()
    def reverse_probs(self, x_t: torch.Tensor, e_t: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Distributions of x_{t-1} and e_{t-1} given (x_t, e_t) at t = step, for step in 1..T.

        Returns (B, N, atom classes) and (B, N, N, bond classes). Each is the
        posterior of noise.py averaged over the denoiser's prediction of the
        clean graph. At step 1 it is that prediction itself.
        """
        x_logits, e_logits = self(x_t, e_t, torch.full((len(x_t),), step, device=x_t.device))
        return (
            previous_probs(x_t, x_logits.softmax(dim=-1), step, self.alpha_bar, self.atom_marginals),
            previous_probs(e_t, e_logits.softmax(dim=-1), step, self.alpha_bar, self.bond_marginals),
        )

    @torch.no_grad()
    def denoise_step(
        self, x_t: torch.Tensor, e_t: torch.Tensor, step: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw (x_{t-1}, e_{t-1}) from `reverse_probs`."""
        x_probs, e_probs = self.reverse_probs(x_t, e_t, step)
        return sample_categorical(x_probs, generator), symmetrize(sample_categorical(e_probs, generator))

    @torch.no_grad()
    def sample(self, num_graphs: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate graphs: start from noise and denoise for T steps.

        Returns (x, e) class indices, (num_graphs, N) and (num_graphs, N, N).
        Turn them into molecules with data.processing.molgraph.decode_il.
        """
        x, e = self.sample_noise(num_graphs, generator)
        for step in range(self.num_steps, 0, -1):
            x, e = self.denoise_step(x, e, step, generator)
        return x, e


def build_model(config: ModelConfig = ModelConfig(), folder: Path = IL_GRAPHS_DIR) -> GraphDiffusion:
    """A model with random weights, sized for the vocabulary and marginals in folder."""
    vocab = Vocab.load(folder / "vocab.json")
    atom_marginals, bond_marginals = train_marginals(folder)
    denoiser = GraphTransformer(
        vocab.num_atom_classes,
        vocab.num_bond_classes,
        hidden_x=config.hidden_x,
        hidden_e=config.hidden_e,
        hidden_t=config.hidden_t,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_multiplier=config.ff_multiplier,
    )
    return GraphDiffusion(
        denoiser, atom_marginals, bond_marginals, vocab.n_max, config.num_steps, config.edge_loss_weight
    )
