"""Signed LIF, surrogate gradients, and reference event convolution kernels."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignedSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, threshold, width):
        ctx.save_for_backward(u, threshold)
        ctx.width = width
        return (u >= threshold).to(u.dtype) - (u <= -threshold).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        u, threshold = ctx.saved_tensors
        width = ctx.width
        g = ((1 - (u - threshold).abs() / width).clamp(min=0)
             + (1 - (u + threshold).abs() / width).clamp(min=0)) / width
        return grad_output * g, None, None

def signed_spike(u, threshold, width=1.0):
    return SignedSpike.apply(u, threshold, width)


"""Reference event kernels: visit nonzero inputs and scatter their contributions.

CPU/PyTorch reference, no dense convolution in these functions. Python dispatch,
indexing, and nonzero scans can dominate wall time. Not a GPU speedup claim.
"""
import torch

@torch.no_grad()
def event_linear(x, weight, out_hw, stride=1, padding=0, transpose=False):
    bsz, cin, _, _ = x.shape
    cout = weight.shape[1] if transpose else weight.shape[0]
    oh, ow = out_hw
    out = x.new_zeros(bsz, cout, oh, ow)
    positions = x.nonzero(as_tuple=False)
    operations = 0
    if not positions.numel():
        return out, operations
    batch, channel, iy, ix = positions.unbind(1)
    values = x[batch, channel, iy, ix]
    # Chunk input events to bound temporary memory (events x output channels).
    flat = out.view(-1)
    out_channels = torch.arange(cout, device=x.device)
    for ky in range(weight.shape[2]):
        for kx in range(weight.shape[3]):
            if transpose:
                oy, ox = iy * stride - padding + ky, ix * stride - padding + kx
                valid = (oy >= 0) & (oy < oh) & (ox >= 0) & (ox < ow)
            else:
                ny, nx = iy + padding - ky, ix + padding - kx
                oy, ox = ny.div(stride, rounding_mode='floor'), nx.div(stride, rounding_mode='floor')
                valid = (ny % stride == 0) & (nx % stride == 0) & (oy >= 0) & (oy < oh) & (ox >= 0) & (ox < ow)
            ids = valid.nonzero(as_tuple=False).flatten()
            for chunk in ids.split(1024):
                if not chunk.numel():
                    continue
                ch = channel[chunk]
                weights = weight[ch, :, ky, kx] if transpose else weight[:, ch, ky, kx].T
                contribution = values[chunk, None] * weights
                indices = (((batch[chunk, None] * cout + out_channels) * oh
                            + oy[chunk, None]) * ow + ox[chunk, None])
                flat.index_add_(0, indices.flatten(), contribution.flatten())
                operations += contribution.numel()
    return out, operations


import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFMixin:
    """Keeps the original D/eta/lambda/group parameters and adds Wy, Wz, a.

    Wz(z) = z - A0 S0 z + B z. A0/S0 use frozen copies of the initial
    dictionary; B is a learned 3x3 correction. Thus at initialization,
    Wy=D0^T, Wz=I-D0^TD0 exactly, including stride and boundary handling.
    D is subsequently trained independently as the synthesis dictionary.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.lams.fill_(1e-4)
            if self.lams_group is not None:
                self.lams_group.fill_(1e-4)
        self.Wy = nn.Parameter(self.dictionary.detach().clone())
        self.Wz = nn.Parameter(torch.zeros(self.out_channels, self.out_channels, 3, 3))
        self.a = nn.Parameter(torch.full((1, self.out_channels, 1, 1), 0.8))
        self.register_buffer('initial_dictionary', self.dictionary.detach().clone())
        self.register_buffer('beta', torch.tensor(0.5))
        self.register_buffer('theta', torch.tensor(0.005))
        self.register_buffer('delta', torch.tensor(0.05))
        self.surrogate_width = 1.0
        self.execution_mode = 'dense'
        self.last_spike_loss = None
        self.last_stats = {}

    def configure_lif(self, options):
        for key in ('beta', 'theta', 'delta'):
            if key in options:
                getattr(self, key).fill_(float(options[key]))
        self.surrogate_width = float(options.get('surrogate_width', 1.0))
        if not 0 <= self.beta.item() <= 1 or self.theta <= 0 or self.delta <= 0 or self.surrogate_width <= 0:
            raise ValueError('Require beta in [0,1], theta/delta/surrogate_width > 0')

    def _geometry(self, y):
        h, w = y.shape[-2:]
        oh = (h + 2*self.padding - self.kernel_size)//self.stride + 1
        ow = (w + 2*self.padding - self.kernel_size)//self.stride + 1
        op = (h - ((oh-1)*self.stride-2*self.padding+self.kernel_size),
              w - ((ow-1)*self.stride-2*self.padding+self.kernel_size))
        return (oh, ow), op

    def forward(self, y):
        if self.execution_mode == 'event' and (self.training or torch.is_grad_enabled()):
            raise RuntimeError('Event mode requires model.eval() and torch.no_grad()')
        hw, op = self._geometry(y)
        z = y.new_zeros(y.shape[0], self.out_channels, *hw)
        v = torch.zeros_like(z)
        drive = F.conv2d(y, self.Wy, stride=self.stride, padding=self.padding)
        feedback, correction = torch.zeros_like(z), torch.zeros_like(z)
        spike_loss = y.new_zeros(())
        event_macs = 0
        event_count = 0
        dense_recurrent = 0
        # Exact non-padding dense contribution count for this geometry.
        # Dense MAC estimate uses standard full-kernel counts (including padding).
        synthesis_macs = z.numel()*self.in_channels*self.kernel_size**2
        analysis_macs = z.numel()*self.in_channels*self.kernel_size**2
        correction_macs = z.numel()*self.out_channels*9
        for t in range(self.T):
            if self.execution_mode == 'dense' and t > 0:
                projected = F.conv_transpose2d(z, self.initial_dictionary,
                    stride=self.stride, padding=self.padding, output_padding=op)
                feedback = F.conv2d(projected, self.initial_dictionary,
                    stride=self.stride, padding=self.padding)
                correction = F.conv2d(z, self.Wz, padding=1)
            q = drive - feedback + correction
            candidate = F.relu(z + F.relu(self.etas[t]) * q - F.relu(self.lams[t]))
            if self.lams_group is not None:
                energy = candidate.norm(p=2, dim=(2,3), keepdim=True) + 1e-8
                candidate = candidate * F.relu(1 - F.relu(self.lams_group[t])/energy)
            u = self.beta*v + self.a*(candidate-z)
            s = signed_spike(u, self.theta, self.surrogate_width)
            dz = self.delta*s
            z = z + dz
            v = u - self.theta*s  # no detach: reset and code paths both participate in BPTT
            spike_loss = spike_loss + s.abs().mean()
            event_count += int(s.detach().count_nonzero())
            if t < self.T-1:
                dense_recurrent += synthesis_macs + analysis_macs + correction_macs
                if self.execution_mode == 'event':
                    image_delta, n1 = event_linear(dz, self.initial_dictionary, y.shape[-2:],
                        self.stride, self.padding, transpose=True)
                    feedback_delta, n2 = event_linear(image_delta, self.initial_dictionary, hw,
                        self.stride, self.padding)
                    correction_delta, n3 = event_linear(dz, self.Wz, hw, padding=1)
                    feedback = feedback + feedback_delta
                    correction = correction + correction_delta
                    event_macs += n1+n2+n3
        recon = F.conv_transpose2d(z, self.dictionary, stride=self.stride,
            padding=self.padding, output_padding=op)
        self.last_spike_loss = spike_loss/self.T
        self.last_stats = {'spikes': event_count, 'opportunities': z.numel()*self.T,
            'spike_rate': event_count/(z.numel()*self.T),
            'dense_recurrent_macs_estimate': dense_recurrent,
            'event_recurrent_macs': event_macs if self.execution_mode == 'event' else None,
            'fixed_dense_macs_estimate': drive.numel()*self.in_channels*self.kernel_size**2+synthesis_macs,
            'mode': self.execution_mode}
        return z, recon

    @torch.no_grad()
    def prox_function(self, threshold=0.0):
        super().prox_function(threshold)
        if hasattr(self, 'a'):
            self.a.clamp_(min=1e-6)



@torch.no_grad()
def select_event_mask(candidate, priority, remaining):
    """Per-image budget; stable flattened-index ordering breaks priority ties.

    If budget is not binding, subthreshold sites keep surrogate gradients.
    If oversubscribed, only the selected sites receive surrogate gradients.
    Selection/ranking itself is deliberately not differentiated.
    """
    mask = torch.zeros_like(candidate)
    for b in range(candidate.shape[0]):
        budget = int(remaining[b])
        if budget <= 0:
            continue
        indices = candidate[b].flatten().nonzero(as_tuple=False).flatten()
        if indices.numel() <= budget:
            mask[b].fill_(1)
        else:
            scores = priority[b].flatten()[indices]
            order = torch.argsort(scores, descending=True, stable=True)
            mask[b].view(-1)[indices[order[:budget]]] = 1
    return mask


class BudgetLIFLayer(nn.Module):
    """Shared Wy/Wz/D/input_gain, signed LIF, per-image total event budget.

    r = Wy*y + Wz*z - z. Dense mode recomputes this expression; event
    inference accumulates Wz*delta_z-delta_z. No LISTA shrinkage is applied.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, T,
                 prox_func_name='no_cons', method='none'):
        super().__init__()
        if T < 1 or kernel_size % 2 != 1 or stride < 1:
            raise ValueError('Require T >= 1, positive stride and odd kernel')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.T = kernel_size, stride, T
        self.padding = (kernel_size - 1) // 2
        self.dictionary = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.Wy = nn.Parameter(self.dictionary.detach().clone())
        self.Wz = nn.Parameter(torch.zeros(out_channels, out_channels, 3, 3))
        self.input_gain = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.8))
        self.register_buffer('beta', torch.tensor(0.9))
        self.register_buffer('theta', torch.tensor(1.0))
        self.register_buffer('delta', torch.tensor(0.1))
        self.surrogate_width = 2.0
        self.event_budget = 1000
        self.execution_mode = 'dense'
        self.last_stats = {}
        self.last_spike_loss = None

    def configure_lif(self, options):
        for key in ('beta', 'theta', 'delta'):
            if key in options:
                getattr(self, key).fill_(float(options[key]))
        self.surrogate_width = float(options.get('surrogate_width', 2.0))
        budget = options.get('event_budget', 1000)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('event_budget must be a nonnegative integer per image/layer')
        self.event_budget = budget
        if not 0 <= self.beta.item() <= 1 or self.theta <= 0 or self.delta <= 0 or self.surrogate_width <= 0:
            raise ValueError('Require beta in [0,1], theta/delta/surrogate_width > 0')

    def forward(self, y):
        if self.execution_mode == 'event' and (self.training or torch.is_grad_enabled()):
            raise RuntimeError('Event mode requires model.eval() and torch.no_grad()')
        drive = F.conv2d(y, self.Wy, stride=self.stride, padding=self.padding)
        z = drive * 0
        v = torch.zeros_like(z)
        signal = drive
        remaining = torch.full((y.shape[0],), self.event_budget, device=y.device, dtype=torch.long)
        used_total = torch.zeros_like(remaining)
        exceeded_steps = torch.zeros_like(remaining)
        rejected_total = torch.zeros_like(remaining)
        event_macs = dense_recurrent = steps = 0
        spike_loss = drive.sum() * 0
        for t in range(self.T):
            if not bool((remaining > 0).any()):
                break
            if self.execution_mode == 'dense' and t > 0:
                signal = drive + F.conv2d(z, self.Wz, padding=1) - z
            potential = self.beta * v + self.input_gain * signal
            candidate = signed_spike(potential, self.theta, self.surrogate_width)
            candidate_count = candidate.detach().ne(0).flatten(1).sum(1)
            # Exhausted images are inactive; do not count hypothetical later spikes.
            excess = (candidate_count - remaining).clamp(min=0) * (remaining > 0)
            exceeded_steps = exceeded_steps + (excess > 0).long()
            rejected_total = rejected_total + excess
            mask = select_event_mask(candidate.detach(), potential.detach().abs() - self.theta, remaining)
            spike = candidate * mask
            used = spike.detach().ne(0).flatten(1).sum(1)
            remaining = remaining - used
            used_total = used_total + used
            dz = self.delta * spike
            z = z + dz
            v = potential - self.theta * spike
            steps += 1
            spike_loss = spike_loss + spike.abs().mean()
            if t == self.T - 1 or not bool((remaining > 0).any()):
                break
            dense_recurrent += z.numel() * self.out_channels * 9
            if self.execution_mode == 'event':
                # Exhausted images will never emit again; skip their feedback.
                active_dz = dz * (remaining > 0).view(-1, 1, 1, 1)
                feedback, count = event_linear(active_dz, self.Wz, z.shape[-2:], padding=1)
                signal = signal + feedback - active_dz
                event_macs += count
        h, w = y.shape[-2:]
        oh, ow = z.shape[-2:]
        output_padding = (h - ((oh-1)*self.stride-2*self.padding+self.kernel_size),
                          w - ((ow-1)*self.stride-2*self.padding+self.kernel_size))
        recon = F.conv_transpose2d(z, self.dictionary, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)
        opportunities = z.numel() * steps
        spikes = int(used_total.sum())
        self.last_spike_loss = spike_loss / max(steps, 1)
        self.last_stats = dict(spikes=spikes, opportunities=opportunities,
            spike_rate=spikes/max(opportunities, 1), steps=steps,
            spikes_per_image=used_total.tolist(), event_budget=self.event_budget,
            budget_exceeded_steps=int(exceeded_steps.sum()),
            budget_rejected_candidates=int(rejected_total.sum()),
            budget_exceeded_steps_per_image=exceeded_steps.tolist(),
            budget_rejected_candidates_per_image=rejected_total.tolist(),
            dense_recurrent_macs_estimate=dense_recurrent,
            event_recurrent_macs=event_macs if self.execution_mode == 'event' else None,
            fixed_dense_macs_estimate=2*z.numel()*self.in_channels*self.kernel_size**2,
            mode=self.execution_mode)
        return z, recon

    @torch.no_grad()
    def prox_function(self, threshold=0.0):
        # Link design: reconstruction loss only, no dictionary shrink/projection.
        pass
