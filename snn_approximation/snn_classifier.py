"""Leaky integrate-and-fire update used by the SNN accelerator."""

import torch


def lif_step(current, membrane, threshold=1.0, decay=0.75):
    membrane = decay * membrane + current
    over_threshold = membrane - threshold
    spike = (over_threshold >= 0).to(current.dtype)
    if torch.is_grad_enabled():
        # Binary forward pass with a smooth surrogate derivative.
        surrogate = over_threshold / (1.0 + 4.0 * over_threshold.abs())
        spike = spike + (surrogate - surrogate.detach())
    return spike, membrane - threshold * spike.detach()
