"""Discrete string diffusion: add noise, compute the training loss, sample.

The string twin of diffusion.py, and the controlled baseline for it: the
same noise process applied to the characters of a SMILES instead of the
atoms and bonds of a graph. `StringDiffusion` wraps a denoiser with the noise
process of noise.py. A string is the (B, L) row of character class indices
described in data/processing/molstring.py.

- `add_noise` corrupts clean strings to a chosen step t.
- `forward` runs the denoiser on noisy strings: logits over the clean classes.
- `loss` is what training would minimise: cross-entropy between those logits
  and the clean string, at a random step per string.
- `sample` starts from pure noise and walks back to step 0, each time drawing
  x_{t-1} from the posterior under the denoiser's prediction.

`build_string_model` creates one with random weights, sized for the
processed data. There is no training loop, as there is none for the graph
model. Both models offer `loss(...)["loss"]` and `sample(n)`, so one loop
can later train either.

Every method that draws random numbers takes an optional torch.Generator for
reproducibility; it must live on the same device as the model.

What is the same as the graph model, because both call noise.py: marginal
transitions, the cosine schedule, T, the posterior used for sampling, and a
plain cross-entropy loss on the predicted clean classes. What differs is the
representation and the denoiser, plus what follows from them:

- A string is one group of variables, so the loss is one term. A graph has
  two (atoms, bonds) and weights the bond term by 5. The two loss values
  therefore cannot be compared with each other; compare the models on their
  samples with optimil2.evaluation.
- There is nothing to mirror, so no step corresponding to `symmetrize`.

Decisions:
- PAD is diffused like any other class, over all L positions. Nothing is
  masked, so a sample's length is whatever number of positions ends up
  non-PAD. The graph model treats EMPTY the same way.
- The loss averages over all positions, PAD included, as the graph loss
  averages over all slots. About 65% of the terms are PAD; whether those
  easy terms drown the informative ones is a question for training, and the
  same question the graph model has.
- A sample is decoded by reading its characters in order and skipping PAD
  (molstring.decode). No grammar is enforced during sampling, so a sample
  need not be a valid SMILES.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from optimil2.data.processing.il_strings import IL_STRINGS_DIR
from optimil2.data.processing.molstring import CharVocab
from optimil2.data.string_dataset import train_marginals
from optimil2.model_primitive.noise import (
    NUM_STEPS,
    cosine_alpha_bar,
    noise_distribution,
    noisy_probs,
    previous_probs,
    sample_categorical,
)
from optimil2.model_primitive.string_denoiser import StringTransformer


@dataclass(frozen=True)
class StringModelConfig:
    num_steps: int = NUM_STEPS  # diffusion steps T
    hidden: int = 128  # width of the per-position vectors
    hidden_t: int = 128  # width of the noise-level embedding
    num_layers: int = 8
    num_heads: int = 8
    ff_multiplier: int = 2  # feed-forward blocks widen by this factor


class StringDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, marginals: torch.Tensor, length: int, num_steps: int = NUM_STEPS):
        """denoiser maps (tokens, t in [0, 1]) to logits; marginals is the noise distribution.

        length is the number of positions L of the strings the marginals were
        measured on. Samples have this many positions: the share of PAD in
        the marginals only means the right string length for this L.
        """
        super().__init__()
        self.denoiser = denoiser
        self.length = length
        self.num_steps = num_steps
        self.register_buffer("marginals", noise_distribution(marginals, "character"))
        self.register_buffer("alpha_bar", cosine_alpha_bar(num_steps).float())

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw x_t from q(. | x0). t is (B,) integer steps in 0..T; step 0 returns the input."""
        return sample_categorical(noisy_probs(x0, self.alpha_bar[t], self.marginals), generator)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Logits over the clean classes for noisy strings at integer steps t, shape (B,).

        Returns (B, L, character classes).
        """
        return self.denoiser(x_t, t / self.num_steps)

    def loss(self, x0: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Cross-entropy of the denoiser's prediction of x0 at one random step per string.

        Returns a dict with the single entry "loss" (to minimise), so that
        it reads like the graph model's loss.
        """
        t = torch.randint(1, self.num_steps + 1, (len(x0),), generator=generator, device=x0.device)
        logits = self(self.add_noise(x0, t, generator), t)
        return {"loss": F.cross_entropy(logits.flatten(0, 1), x0.flatten())}

    def sample_noise(self, num_strings: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw strings of pure noise: every position independently from the marginals."""
        return sample_categorical(self.marginals.expand(num_strings, self.length, -1), generator)

    @torch.no_grad()
    def reverse_probs(self, x_t: torch.Tensor, step: int) -> torch.Tensor:
        """Distribution of x_{t-1} given x_t at t = step, for step in 1..T.

        Returns (B, L, character classes): the posterior of noise.py averaged
        over the denoiser's prediction of the clean string. At step 1 it is
        that prediction itself.
        """
        logits = self(x_t, torch.full((len(x_t),), step, device=x_t.device))
        return previous_probs(x_t, logits.softmax(dim=-1), step, self.alpha_bar, self.marginals)

    @torch.no_grad()
    def denoise_step(self, x_t: torch.Tensor, step: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw x_{t-1} from `reverse_probs`."""
        return sample_categorical(self.reverse_probs(x_t, step), generator)

    @torch.no_grad()
    def sample(self, num_strings: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate strings: start from noise and denoise for T steps.

        Returns (num_strings, L) class indices. Turn each row into a SMILES
        with data.processing.molstring.decode.
        """
        x = self.sample_noise(num_strings, generator)
        for step in range(self.num_steps, 0, -1):
            x = self.denoise_step(x, step, generator)
        return x


def build_string_model(config: StringModelConfig = StringModelConfig(), folder: Path = IL_STRINGS_DIR) -> StringDiffusion:
    """A model with random weights, sized for the vocabulary and marginals in folder."""
    vocab = CharVocab.load(folder / "vocab.json")
    denoiser = StringTransformer(
        vocab.num_classes,
        vocab.l_max,
        hidden=config.hidden,
        hidden_t=config.hidden_t,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_multiplier=config.ff_multiplier,
    )
    return StringDiffusion(denoiser, train_marginals(folder), vocab.l_max, config.num_steps)
