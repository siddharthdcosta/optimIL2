"""Noise for discrete graph diffusion: schedule, transition matrices, posterior.

This follows DiGress (Vignac et al., "DiGress: Discrete Denoising Diffusion
for Graph Generation", ICLR 2023). Every atom slot and every slot pair holds
one class out of K. One step of the forward process keeps a class with
probability alpha_t and otherwise redraws it from a fixed distribution m:

    Q_t = alpha_t I + (1 - alpha_t) 1 m^T        Q_t[a, b] = q(x_t = b | x_{t-1} = a)

Steps 1..t applied in a row have the same form, with
alpha_bar_t = alpha_1 * ... * alpha_t:

    Qbar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 m^T

so x_t can be drawn from x_0 in one go, and x_T is distributed as m whatever
x_0 was. Sampling runs this backwards with the posterior below.

Decisions:
- m is the class frequency of the training set, not uniform (DiGress's
  "marginal transitions"). About 60% of slots are EMPTY and 98.6% of pairs
  are NO_BOND, so noise drawn from m already looks like a sparse graph of
  the right size, while uniform noise would be a dense graph of 56 atoms.
- Cosine schedule for alpha_bar (Nichol & Dhariwal 2021), as in DiGress.
- Steps are numbered 1..T and x_0 is the data itself (alpha_bar_0 = 1).
- Every class must have m > 0. Then every entry of Qbar_t is positive for
  t >= 1 and the posterior never divides by zero. The data pipeline
  guarantees this and the diffusion wrapper checks it.
"""

import math

import torch


def cosine_alpha_bar(num_steps: int, offset: float = 0.008) -> torch.Tensor:
    """alpha_bar_t for t = 0..num_steps, shape (num_steps + 1,), float64.

    Starts at exactly 1 (no noise) and falls to 0 (pure noise) along a
    squared cosine.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    curve = torch.cos((steps / num_steps + offset) / (1 + offset) * math.pi / 2) ** 2
    return curve / curve[0]


def transition(keep: torch.Tensor, marginals: torch.Tensor) -> torch.Tensor:
    """Transition matrices keep * I + (1 - keep) * 1 m^T, one per batch element.

    keep is (B,), marginals is (K,); the result is (B, K, K). Row a is the
    distribution of the class after the transition, given class a before it.
    """
    num_classes = marginals.numel()
    identity = torch.eye(num_classes, dtype=marginals.dtype, device=marginals.device)
    keep = keep.to(marginals.dtype).view(-1, 1, 1)
    return keep * identity + (1 - keep) * marginals.view(1, 1, num_classes)


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
    variables (slots, or slot pairs flattened). q_step, q_bar_prev and q_bar
    are Q_t, Qbar_{t-1} and Qbar_t, each (B, K, K). Returns (B, M, K).
    """
    batch = torch.arange(x_t.shape[0], device=x_t.device).unsqueeze(1)
    likelihood = q_step.transpose(1, 2)[batch, x_t]  # [b, m, k] = Q_t[k, x_t]
    evidence = q_bar.transpose(1, 2)[batch, x_t]  # [b, m, j] = Qbar_t[j, x_t]
    probs = likelihood * ((p_x0 / evidence) @ q_bar_prev)
    return probs / probs.sum(dim=-1, keepdim=True)


def sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one class index per distribution. probs is (..., K); returns (...)."""
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])


def symmetrize(e: torch.Tensor) -> torch.Tensor:
    """Copy the upper triangle of (B, N, N) class indices onto the lower one and zero the diagonal."""
    upper = torch.triu(e, diagonal=1)
    return upper + upper.transpose(1, 2)
