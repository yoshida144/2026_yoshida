import copy

import torch

import torch.nn as nn

import torch.nn.functional as F

from prox_func import on_l2_sphere_softshrink, on_l2_sphere_relu, group_lasso_in_l2_sphere, group_lasso_on_l2_sphere, no_constraints, in_box_softshrink, on_l2_sphere

torch.backends.cudnn.benchmark = True

class LISTALayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, T, prox_func_name = "st_on_sphere", method = "group"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.T = T
        
        self.padding = (kernel_size - 1) // 2
        
        # Initialization
        self.dictionary = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        
        self.etas = nn.Parameter(torch.ones(T) * 0.1)
        self.lams = nn.Parameter(torch.ones(T) * 0.01)
        if method == "group":
            self.lams_group = nn.Parameter(torch.ones(T) * 0.01)
        else:
            self.lams_group = None
        
        if prox_func_name == 'st_on_sphere':
            prox_func_dictionary = on_l2_sphere_softshrink
        elif prox_func_name == 'in_box_st':
            prox_func_dictionary = in_box_softshrink
        elif prox_func_name == 'on_sphere':
            prox_func_dictionary = on_l2_sphere
            on_l2_sphere(dictionary=self.dictionary)
        elif prox_func_name == 'nn_on_sphere':
            prox_func_dictionary = on_l2_sphere_relu
        elif prox_func_name == 'no_cons':
            prox_func_dictionary = no_constraints
            self.dictionary = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        elif prox_func_name == 'group_in_sphere':
            prox_func_dictionary = group_lasso_in_l2_sphere
            on_l2_sphere(dictionary=self.dictionary)
        elif prox_func_name == 'group_on_sphere':
            prox_func_dictionary = group_lasso_on_l2_sphere
            on_l2_sphere(dictionary=self.dictionary)
        
        self.prox_func_dictionary = prox_func_dictionary
        
        with torch.no_grad():
            self.prox_function()
            
        self.register_buffer('selective_probability', torch.zeros(out_channels))

    def forward(self, y, *, return_reconstruction=True, trace=None, predict_penultimate=None, skipped_updates=1):
        batch_size = y.shape[0]
        h, w = y.shape[2], y.shape[3]
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        x = torch.zeros(batch_size, self.out_channels, out_h, out_w, device=y.device)
        
        kernel_h, kernel_w = self.dictionary.shape[2], self.dictionary.shape[3]
        stride_h, stride_w = self.stride, self.stride
        padding_h, padding_w = self.padding, self.padding
        
        outpadding_h = h - ((out_h - 1) * stride_h - 2 * padding_h + (kernel_h - 1) + 1)
        outpadding_w = w - ((out_w - 1) * stride_w - 2 * padding_w + (kernel_w - 1) + 1)
        
        first_skipped = self.T - 1 - skipped_updates
        if predict_penultimate is not None and not 1 <= skipped_updates <= self.T - 2:
            raise ValueError("Keep at least the first and last exact LISTA updates.")
        previous = x
        for t in range(self.T):
            if predict_penultimate is not None:
                if first_skipped <= t < self.T - 1:
                    if t == self.T - 2:
                        x = predict_penultimate(previous, x)
                    if trace is not None:
                        trace.append(x)
                    continue
                if t == first_skipped - 1:
                    previous = x
            eta = F.relu(self.etas[t])
            lam = F.relu(self.lams[t])
            
            if self.lams_group is not None:
                lam_group = F.relu(self.lams_group[t])
            
            recon = F.conv_transpose2d(
                x, self.dictionary,
                stride=self.stride,
                padding=self.padding,
                output_padding=(outpadding_h, outpadding_w)
            )
            
            residual = recon - y
            
            grad = F.conv2d(
                residual, self.dictionary,
                stride=self.stride,
                padding=self.padding
            )
            
            z = x - eta * grad
            
            z = torch.relu(z - lam)
            
            if self.lams_group is not None:
                channel_energy = torch.norm(z, p = 2, dim=(2,3), keepdim=True)
                channel_energy = channel_energy + 1e-8

                group_scale = F.relu(1 - lam_group / channel_energy)
                
                z = z * group_scale

            x = z
            if trace is not None:
                trace.append(x)
            
        if not return_reconstruction:
            return x, None

        recon = F.conv_transpose2d(
            x, self.dictionary,
            stride=self.stride,
            padding=self.padding,
            output_padding=(outpadding_h, outpadding_w)
        )
        return x, recon

    @torch.no_grad()
    def prox_function(self, threshold=0.0):
        lams = self.lams.data
        lams.clamp_(min=1e-6)
        etas = self.etas.data
        etas.clamp_(min=1e-6)
        if self.lams_group is not None:
            lams_group = self.lams_group.data
            lams_group.clamp_(min=1e-6)
        
        self.dictionary.data = self.prox_func_dictionary(self.dictionary.data, threshold)

class LISTA_ML_CSC(nn.Module):
    def __init__(self, input_channels, num_classes, layer_specs, method = "group"):
        super().__init__()
        
        current_in = input_channels
        self.layers = nn.ModuleList()
        self.layer_specs = copy.deepcopy(layer_specs)
        
        for spec in layer_specs:
            
            layer = LISTALayer(
                in_channels=current_in,
                out_channels=spec['dim'],
                kernel_size=spec['kernel'],
                stride=spec['stride'],
                T=spec['T'],
                prox_func_name=spec['prox'],
                method=method
            )
            self.layers.append(layer)
            current_in = spec['dim']

    def forward(self, images, *, return_reconstructions=True):
        current_input = images
        codes = []
        recons = []
        
        for layer in self.layers:
            code, recon = layer(current_input, return_reconstruction=return_reconstructions)
            codes.append(code)
            recons.append(recon)
            current_input = code
        
        final_code = codes[-1]
        return final_code, codes, recons

class ML_CSC_Image_Classification(nn.Module):
    def __init__(self, input_channels, num_classes, layer_specs, method = "group"):
        super().__init__()
        
        self.feature_extractor = LISTA_ML_CSC(input_channels, num_classes, layer_specs, method = method)
            
        self.final_classifier = nn.Linear(self.feature_extractor.layers[-1].out_channels, num_classes, bias=False)

    def forward(self, images, *, return_reconstructions=True):
        final_code, codes, recons = self.feature_extractor(
            images, return_reconstructions=return_reconstructions
        )
        
        vec_features = torch.amax(final_code, dim=(2, 3))
        
        logits = self.final_classifier(vec_features)
        
        return logits, codes, recons
