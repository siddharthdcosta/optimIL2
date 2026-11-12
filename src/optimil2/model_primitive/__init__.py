"""A first, deliberately crude generative model for ionic liquids: discrete graph diffusion.

- noise.py      the forward (corruption) process and the posterior used to sample
- denoiser.py   the network that predicts the clean graph from a noisy one
- diffusion.py  ties the two together: add noise, compute the loss, sample
- checks.py     self-checks, run by scripts/model/smoke_test.py

There is no training loop yet.
"""
