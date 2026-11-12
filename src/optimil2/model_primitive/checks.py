"""Self-checks for the crude graph diffusion model, run by scripts/model/smoke_test.py.

The string model has the same checks in string_checks.py. `check_noise_math`
and `known_x0_posterior` test the shared noise process and are used by both.

Nothing here trains. Each check raises AssertionError with a message when it
fails and otherwise returns a one-line summary. They cover what cannot be
seen by reading shapes alone:

- the transition matrices and the posterior, against brute-force arithmetic
- the forward process, against the distribution it should produce
- the denoiser: shapes, symmetry, and independence of slot order
- the loss: value at initialisation, and that every weight gets a gradient
- the reverse process, with an "oracle" in place of the network (it always
  predicts the true clean graph): every step must then be the exact
  posterior, and sampling must end exactly on that graph
- sampling with the real, untrained network, end to end, scored with the
  metrics of optimil2.evaluation
"""

import math

import torch
from torch import nn

from optimil2.data.processing.molgraph import Vocab, to_smiles
from optimil2.evaluation import evaluate
from optimil2.model_primitive.diffusion import GraphDiffusion
from optimil2.model_primitive.noise import cosine_alpha_bar, posterior, transition


def check_noise_math(marginals: torch.Tensor, num_steps: int = 500) -> str:
    """Closed-form transitions and the posterior agree with brute-force arithmetic (float64)."""
    m = marginals.double() / marginals.double().sum()
    k = m.numel()
    alpha_bar = cosine_alpha_bar(num_steps)
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    assert alpha_bar[0] == 1 and alpha_bar[-1] < 1e-6 and (alpha_bar.diff() < 0).all(), "alpha_bar must fall from 1 to 0"

    product = torch.eye(k, dtype=torch.float64)
    closed_form_error = 0.0
    for t in range(1, num_steps + 1):
        product = product @ transition(alpha[t - 1 : t], m)[0]
        closed_form_error = max(closed_form_error, (product - transition(alpha_bar[t : t + 1], m)[0]).abs().max().item())
    assert closed_form_error < 1e-10, f"Q_1...Q_t differs from Qbar_t by {closed_form_error}"

    x_t = torch.arange(k).view(1, k)  # variable i currently has class i
    generator = torch.Generator().manual_seed(0)
    posterior_error = mixture_error = 0.0
    for t in (1, 2, num_steps // 2, num_steps - 1, num_steps):
        q_step, q_bar_prev, q_bar = (transition(a.view(1), m) for a in (alpha[t - 1], alpha_bar[t - 1], alpha_bar[t]))
        # When x_0 = j is known:  sum over x_t of q(x_t | x_0) q(x_{t-1} | x_t, x_0)  must equal  q(x_{t-1} | x_0).
        known = torch.zeros(k, k, k, dtype=torch.float64)  # [j, i, :] = q(x_{t-1} | x_t = i, x_0 = j)
        for j in range(k):
            certain = torch.zeros(1, k, k, dtype=torch.float64)
            certain[0, :, j] = 1
            known[j] = posterior(x_t, certain, q_step, q_bar_prev, q_bar)[0]
            posterior_error = max(posterior_error, (q_bar[0, j] @ known[j] - q_bar_prev[0, j]).abs().max().item())
        # When x_0 is uncertain, the result must be the average of the known cases, weighted by p(x_0 = j).
        p_x0 = torch.rand(1, k, k, dtype=torch.float64, generator=generator)
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)
        mixed = torch.einsum("ij,jik->ik", p_x0[0], known)
        mixture_error = max(mixture_error, (posterior(x_t, p_x0, q_step, q_bar_prev, q_bar)[0] - mixed).abs().max().item())
    assert posterior_error < 1e-10, f"posterior is inconsistent with the forward process by {posterior_error}"
    assert mixture_error < 1e-10, f"posterior under an uncertain x_0 is not the weighted average, off by {mixture_error}"
    return f"transitions and posterior exact (errors {closed_form_error:.0e}, {posterior_error:.0e}, {mixture_error:.0e})"


def known_x0_posterior(
    noisy: torch.Tensor, clean: torch.Tensor, marginals: torch.Tensor, step: int, alpha_bar: torch.Tensor
) -> torch.Tensor:
    """q(x_{t-1} | x_t, x_0) at t = step for every variable, straight from Bayes' rule in float64 on the CPU.

    noisy and clean are class indices of the same shape (x_t and x_0) and
    alpha_bar is the float64 schedule. Returns that shape plus a class axis.
    This is what a reverse step must produce when the denoiser is certain of
    x_0. It is written entry by entry, without the functions under test.
    """
    noisy, clean, m = noisy.cpu(), clean.cpu(), marginals.cpu().double()
    q_step, q_bar_prev, q_bar = (
        transition(keep.view(1), m)[0] for keep in (alpha_bar[step] / alpha_bar[step - 1], alpha_bar[step - 1], alpha_bar[step])
    )
    # [..., k] = q(x_{t-1} = k | x_0) * q(x_t | x_{t-1} = k) / q(x_t | x_0)
    return q_bar_prev[clean] * q_step.T[noisy] / q_bar[clean, noisy].unsqueeze(-1)


def check_add_noise(model: GraphDiffusion, x0: torch.Tensor, e0: torch.Tensor, repeats: int = 16) -> str:
    """Noised graphs are valid graphs, and agree with x_0 as often as the schedule says.

    The batch is repeated to get enough variables for a tight statistical
    test: the tolerance is 5 standard errors, so a correct model never fails.
    """
    x0, e0 = x0.repeat(repeats, 1), e0.repeat(repeats, 1, 1)
    steps = torch.full((len(x0),), 0, device=x0.device)
    x_t, e_t = model.add_noise(x0, e0, steps)
    assert torch.equal(x_t, x0) and torch.equal(e_t, e0), "step 0 must return the clean graph"

    rows, cols = torch.triu_indices(e0.shape[1], e0.shape[1], offset=1, device=e0.device)
    worst = 0.0
    for step in (1, model.num_steps // 2, model.num_steps):
        x_t, e_t = model.add_noise(x0, e0, steps + step)
        assert torch.equal(e_t, e_t.transpose(1, 2)), "noisy bonds must be symmetric"
        assert not e_t.diagonal(dim1=1, dim2=2).any(), "noisy bonds must have a NO_BOND diagonal"
        keep = model.alpha_bar[step].item()
        pairs = ((x_t, x0, model.atom_marginals), (e_t[:, rows, cols], e0[:, rows, cols], model.bond_marginals))
        for noisy, clean, marginals in pairs:
            # A variable equals its clean value with probability keep + (1 - keep) * m[clean value].
            expected = (keep + (1 - keep) * marginals[clean]).mean().item()
            observed = (noisy == clean).float().mean().item()
            tolerance = 5 * math.sqrt(0.25 / clean.numel()) + 1e-4
            assert abs(observed - expected) < tolerance, f"step {step}: {observed:.4f} unchanged, expected {expected:.4f}"
            worst = max(worst, abs(observed - expected))
    return f"forward process matches the schedule (largest deviation {worst:.4f})"


def check_denoiser(model: GraphDiffusion, x: torch.Tensor, e: torch.Tensor) -> str:
    """Output shapes, finiteness, symmetric bond logits, and independence of slot order."""
    batch, slots = x.shape
    t = torch.randint(1, model.num_steps + 1, (batch,), device=x.device)
    with torch.no_grad():
        x_logits, e_logits = model(x, e, t)
        assert x_logits.shape == (batch, slots, model.atom_marginals.numel()), f"atom logits have shape {tuple(x_logits.shape)}"
        assert e_logits.shape == (batch, slots, slots, model.bond_marginals.numel()), f"bond logits have shape {tuple(e_logits.shape)}"
        assert x_logits.isfinite().all() and e_logits.isfinite().all(), "logits are not finite"
        assert torch.equal(e_logits, e_logits.transpose(1, 2)), "bond logits must be symmetric"

        order = torch.randperm(slots, device=x.device)
        x_permuted, e_permuted = model(x[:, order], e[:, order][:, :, order], t)
    error = max(
        (x_permuted - x_logits[:, order]).abs().max().item(),
        (e_permuted - e_logits[:, order][:, :, order]).abs().max().item(),
    )
    assert error < 1e-3, f"permuting the slots changes the output by {error}"
    return f"logits {tuple(x_logits.shape)} and {tuple(e_logits.shape)}, slot-order error {error:.0e}"


def check_loss(model: GraphDiffusion, x0: torch.Tensor, e0: torch.Tensor) -> str:
    """The loss starts near log(number of classes), and every weight receives a gradient."""
    model.zero_grad()
    parts = model.loss(x0, e0)
    parts["loss"].backward()
    for name, num_classes in (("node", model.atom_marginals.numel()), ("edge", model.bond_marginals.numel())):
        value, uniform = parts[name].item(), math.log(num_classes)
        assert 0.5 * uniform < value < 1.5 * uniform, f"{name} loss {value:.3f} is far from log({num_classes}) = {uniform:.3f}"
    dead = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.isfinite().all() or not p.grad.any()]
    assert not dead, f"parameters with no (or non-finite) gradient: {dead}"
    model.zero_grad()
    return f"loss {parts['loss'].item():.3f} (node {parts['node'].item():.3f}, edge {parts['edge'].item():.3f}); all weights get gradients"


class Oracle(nn.Module):
    """Stands in for the denoiser and always predicts a known clean graph with certainty."""

    def __init__(self, x0: torch.Tensor, e0: torch.Tensor, num_atom_classes: int, num_bond_classes: int):
        super().__init__()
        self.x_logits = 50.0 * nn.functional.one_hot(x0, num_atom_classes).float()
        self.e_logits = 50.0 * nn.functional.one_hot(e0, num_bond_classes).float()

    def forward(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_logits, self.e_logits


def check_reverse_process(model: GraphDiffusion, x0: torch.Tensor, e0: torch.Tensor, num_steps: int = 50) -> str:
    """With an oracle denoiser, each reverse step is the exact posterior and sampling ends on the clean graph.

    If the network knows x_0, the distribution `reverse_probs` returns for a
    variable must be q(x_{t-1} | x_t, x_0). That is recomputed by
    `known_x0_posterior` for graphs noised to step t. This tests how
    `reverse_probs` is wired (which matrices, which step), not only the
    posterior formula.
    """
    oracle = Oracle(x0, e0, model.atom_marginals.numel(), model.bond_marginals.numel())
    chain = GraphDiffusion(oracle, model.atom_marginals, model.bond_marginals, model.num_slots, num_steps).to(x0.device)
    alpha_bar = cosine_alpha_bar(num_steps)

    worst = 0.0
    for step in (1, 2, num_steps // 2, num_steps - 1, num_steps):
        x_t, e_t = chain.add_noise(x0, e0, torch.full((len(x0),), step, device=x0.device))
        x_probs, e_probs = chain.reverse_probs(x_t, e_t, step)
        for probs, noisy, clean, marginals in ((x_probs, x_t, x0, chain.atom_marginals), (e_probs, e_t, e0, chain.bond_marginals)):
            error = (probs.cpu().double() - known_x0_posterior(noisy, clean, marginals, step, alpha_bar)).abs().max().item()
            assert error < 1e-4, f"step {step}: reverse distribution differs from Bayes' rule by {error}"
            worst = max(worst, error)

    x, e = chain.sample(len(x0))
    assert torch.equal(x, x0) and torch.equal(e, e0), "with an oracle denoiser the sample must equal the clean graph"
    return f"reverse steps match Bayes' rule (error {worst:.0e}); oracle sampling recovers all {len(x0)} graphs exactly"


def check_sampling(model: GraphDiffusion, vocab: Vocab, train_smiles: list[str], num_graphs: int = 4) -> str:
    """Full sampling with the untrained network gives well-formed graph tensors.

    The samples are scored against train_smiles with optimil2.evaluation,
    the same call that scores the string model. Random weights should not
    produce valid ionic liquids; the metrics in the summary are a baseline
    for a trained model, not a pass criterion.
    """
    x, e = model.sample(num_graphs)
    assert x.shape == (num_graphs, model.num_slots) and e.shape == (num_graphs, model.num_slots, model.num_slots)
    assert x.dtype == torch.long and e.dtype == torch.long
    assert 0 <= x.min() and x.max() < vocab.num_atom_classes, "atom class out of range"
    assert 0 <= e.min() and e.max() < vocab.num_bond_classes, "bond class out of range"
    assert torch.equal(e, e.transpose(1, 2)) and not e.diagonal(dim1=1, dim2=2).any(), "sampled bonds must be symmetric"
    metrics = evaluate([to_smiles(x[i], e[i], vocab) for i in range(num_graphs)], train_smiles)
    atoms = (x != 0).sum(dim=1).float().mean().item()
    return f"{num_graphs} graphs sampled in {model.num_steps} steps, {atoms:.0f} atoms on average; {metrics} (untrained)"
