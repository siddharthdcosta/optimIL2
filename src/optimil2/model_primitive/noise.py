"""Noise for discrete diffusion: schedule, transition matrices, posterior.

This follows DiGress (Vignac et al., "DiGress: Discrete Denoising Diffusion
for Graph Generation", ICLR 2023). The data is a set of categorical
variables, each holding one class out of K: the atom slots and the slot
pairs of a graph, or the character positions of a string. Nothing here knows
which. Both models (diffusion.py for graphs, string_diffusion.py for
strings) are built from these functions, so they share one noise process.

One step of the forward process keeps a class with probability alpha_t and
otherwise redraws it from a fixed distribution m:

    Q_t = alpha_t I + (1 - alpha_t) 1 m^T        Q_t[a, b] = q(x_t = b | x_{t-1} = a)

Steps 1..t applied in a row have the same form, with
alpha_bar_t = alpha_1 * ... * alpha_t:

    Qbar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 m^T

so x_t can be drawn from x_0 in one go, and x_T is distributed as m whatever
x_0 was. Sampling runs this backwards with the posterior below.

Decisions:
- m is the class frequency of the training set, not uniform (DiGress's
  "marginal transitions"). About 60% of atom slots are EMPTY, 98.6% of pairs
  are NO_BOND and 65% of string positions are PAD, so noise drawn from m
  already has roughly the right size: a sparse graph, or a short string.
  Uniform noise would be a dense graph of 56 atoms.
- Cosine schedule for alpha_bar (Nichol & Dhariwal 2021), as in DiGress.
- T = 500 steps (NUM_STEPS), the DiGress setting for molecules, for every
  model.
- Steps are numbered 1..T and x_0 is the data itself (alpha_bar_0 = 1).
- Every class must have m > 0. Then every entry of Qbar_t is positive for
  t >= 1 and the posterior never divides by zero. The data pipelines
  guarantee this and `noise_distribution` checks it.
- No absorbing ("mask") noise, where a variable turns into a special MASK
  class instead of being redrawn from m. That is this same process with all
  of m on MASK, so every real class would have m = 0 and break the rule
  above. Supporting it means a guard against dividing by zero in the
  posterior, suppressing the network's MASK output, and a check for each.
  That is not free, so it is left out, even as an option.
"""

import math

import torch

NUM_STEPS = 500  # diffusion steps T


def cosine_alpha_bar(num_steps: int, offset: float = 0.008) -> torch.Tensor:
    """alpha_bar_t for t = 0..num_steps, shape (num_steps + 1,), float64.

    Starts at exactly 1 (no noise) and falls to 0 (pure noise) along a
    squared cosine.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    curve = torch.cos((steps / num_steps + offset) / (1 + offset) * math.pi / 2) ** 2
    return curve / curve[0]


def noise_distribution(frequencies: torch.Tensor, name: str) -> torch.Tensor:
    """Turn class frequencies into the noise distribution m: float32, sums to 1.

    Raises ValueError unless every class has a positive frequency. name says
    which classes these are, for the error message.
    """
    if frequencies.min() <= 0:
        raise ValueError(f"every {name} class needs a positive marginal, got {frequencies.tolist()}")
    return (frequencies / frequencies.sum()).float()


def transition(keep: torch.Tensor, marginals: torch.Tensor) -> torch.Tensor:
    """Transition matrices keep * I + (1 - keep) * 1 m^T, one per batch element.

    keep is (B,), marginals is (K,); the result is (B, K, K). Row a is the
    distribution of the class after the transition, given class a before it.
    """
    num_classes = marginals.numel()
    identity = torch.eye(num_classes, dtype=marginals.dtype, device=marginals.device)
    keep = keep.to(marginals.dtype).view(-1, 1, 1)
    return keep * identity + (1 - keep) * marginals.view(1, 1, num_classes)


def noisy_probs(x0: torch.Tensor, keep: torch.Tensor, marginals: torch.Tensor) -> torch.Tensor:
    """Distribution of x_t given x_0: each variable's row of Qbar_t.

    x0 is (B, ...) class indices, with any number of variables per batch
    element in any layout. keep is (B,), the alpha_bar_t of each element.
    Returns (B, ..., K).
    """
    batch = torch.arange(len(x0), device=x0.device).view(-1, *[1] * (x0.dim() - 1))
    return transition(keep, marginals)[batch, x0]


def posterior(
    x_t: torch.Tensor,
    p_x0: torch.Tensor,
    q_step: torch.Tensor,
    q_bar_prev: torch.Tensor,
    q_bar: torch.Tensor,
) -> torch.Tensor:
    """Distribution of x_{t-1} given x_t and a predicted distribution over x_0.

        p(x_{t-1} = k | x_t) = sum_j  p_x0[j] * q(x_{t-1} = k | x_t, x_0 = j)

    with, by Bayes' rule,

        q(x_{t-1} = k | x_t, x_0 = j) = Qbar_{t-1}[j, k] * Q_t[k, x_t] / Qbar_t[j, x_t]

    x_t is (B, M) class indices and p_x0 is (B, M, K), where M counts the
    variables (slots, slot pairs flattened, or string positions). q_step,
    q_bar_prev and q_bar are Q_t, Qbar_{t-1} and Qbar_t, each (B, K, K).
    Returns (B, M, K).
    """
    batch = torch.arange(x_t.shape[0], device=x_t.device).unsqueeze(1)
    likelihood = q_step.transpose(1, 2)[batch, x_t]  # [b, m, k] = Q_t[k, x_t]
    evidence = q_bar.transpose(1, 2)[batch, x_t]  # [b, m, j] = Qbar_t[j, x_t]
    probs = likelihood * ((p_x0 / evidence) @ q_bar_prev)
    return probs / probs.sum(dim=-1, keepdim=True)


def previous_probs(
    x_t: torch.Tensor, p_x0: torch.Tensor, step: int, alpha_bar: torch.Tensor, marginals: torch.Tensor
) -> torch.Tensor:
    """One reverse step: the distribution of x_{t-1} given x_t at t = step, for step in 1..T.

    This is `posterior` with the three matrices of that step filled in from
    the schedule. x_t is (B, ...) class indices in any layout and p_x0 is
    (B, ..., K), the predicted distribution over the clean classes.
    alpha_bar is the (T + 1,) schedule. Returns (B, ..., K). At step 1 the
    result is p_x0 itself.
    """
    t = torch.full((len(x_t),), step, device=x_t.device)
    keep_bar, keep_bar_prev = alpha_bar[t], alpha_bar[t - 1]
    keep = keep_bar / keep_bar_prev  # alpha_t of this single step
    probs = posterior(
        x_t.flatten(1),
        p_x0.flatten(1, -2),
        transition(keep, marginals),
        transition(keep_bar_prev, marginals),
        transition(keep_bar, marginals),
    )
    return probs.view(*x_t.shape, -1)


def sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one class index per distribution. probs is (..., K); returns (...)."""
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])


def symmetrize(e: torch.Tensor) -> torch.Tensor:
    """Copy the upper triangle of (B, N, N) class indices onto the lower one and zero the diagonal."""
    upper = torch.triu(e, diagonal=1)
    return upper + upper.transpose(1, 2)
