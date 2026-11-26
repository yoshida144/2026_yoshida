import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

DictionaryProx = Callable[[torch.Tensor, float], torch.Tensor]

@torch.no_grad()
def on_l2_sphere(dictionary: torch.Tensor, threshold: float = 0.0):
    d = dictionary.data
    norms = d.view(d.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    norms.clamp_(min=1e-6)
    d.div_(norms + 1e-6)
    return dictionary

@torch.no_grad()
def in_l2_sphere(dictionary: torch.Tensor, threshold: float = 0.0):
    d = dictionary.data
    norms = d.view(d.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    norms.clamp_(min=1.0)
    d.div_(norms)
    return dictionary

@torch.no_grad()
def on_l2_sphere_softshrink(dictionary: torch.Tensor, threshold: float):
    dictionary.data = F.softshrink(dictionary.data, threshold)
    on_l2_sphere(dictionary)
    return dictionary

@torch.no_grad()
def on_l2_sphere_relu(dictionary: torch.Tensor, threshold: float):
    dictionary.data.sub_(threshold)
    dictionary.data = F.relu(dictionary.data)
    on_l2_sphere(dictionary)
    return dictionary

@torch.no_grad()
def group_lasso_in_l2_sphere(dictionary: torch.Tensor, threshold: float, eps: float = 1e-6):
    d = dictionary.data
    out_ch, in_ch, k, _ = d.shape
    
    connection_norms = d.view(out_ch, in_ch, -1).norm(p=2, dim=2).view(out_ch, in_ch, 1, 1)
    
    shrinkage_factor = 1.0 - threshold / (connection_norms + eps)
    shrinkage_factor = torch.relu(shrinkage_factor)
    
    d.mul_(shrinkage_factor)
    filter_norms = d.view(out_ch, -1).norm(p=2, dim=1).view(out_ch, 1, 1, 1)
    
    scale_denominator = torch.clamp(filter_norms, min=1.0)
    
    d.div_(scale_denominator)

    return dictionary

@torch.no_grad()
def group_lasso_on_l2_sphere(dictionary: torch.Tensor, threshold: float, eps: float = 1e-12):
    
    d = dictionary.data
    out_ch, in_ch, k, _ = d.shape

    connection_norms = d.view(out_ch, in_ch, -1).norm(p=2, dim=2).view(out_ch, in_ch, 1, 1)
    
    shrinkage_factor = 1.0 - threshold / (connection_norms + eps)
    shrinkage_factor = torch.relu(shrinkage_factor)
    
    d.mul_(shrinkage_factor)
    
    filter_norms = d.view(out_ch, -1).norm(p=2, dim=1).view(out_ch, 1, 1, 1)
    
    div_factor = torch.where(
        filter_norms > eps,
        1.0 / (filter_norms + eps), 
        torch.zeros_like(filter_norms)
    )
    
    d.mul_(div_factor)

    return dictionary

@torch.no_grad()
def in_box(dictionary, threshold=1.0):
    radius = 1.0 
    return torch.clamp(dictionary, min=-0.0, max=radius)

@torch.no_grad()
def in_box_relu(dictionary: torch.Tensor, threshold: float):

    if threshold == 0.0:
        in_box(dictionary)
    else:
        dictionary.data.sub_(threshold)
        dictionary.data = F.relu(dictionary.data)

    return dictionary

@torch.no_grad()
def in_l2_sphere_softshrink(dictionary: torch.Tensor, threshold: float):
    if threshold == 0.0:
        in_l2_sphere(dictionary)
    else:
        dictionary.data = F.softshrink(dictionary.data, threshold)

    return dictionary

@torch.no_grad()
def in_l2_sphere_relu(dictionary: torch.Tensor, threshold: float):
    dictionary.data = F.relu(dictionary.data)

    if threshold == 0.0:
        in_l2_sphere(dictionary)

    return dictionary

@torch.no_grad()
def in_box_softshrink(dictionary: torch.Tensor, threshold: float):
    if threshold == 0.0:
        dictionary.data.clamp_(min=-1.0, max=1.0)
    else:
        dictionary.data = F.softshrink(dictionary.data, threshold)

    return dictionary

@torch.no_grad()
def no_constraints_softshrink(dictionary: torch.Tensor, threshold: float):
    if threshold == 0.0:
        pass
    else:
        dictionary.data = F.softshrink(dictionary.data, threshold)

    return dictionary

@torch.no_grad()
def no_constraints(dictionary: torch.Tensor, threshold: float):
    return dictionary