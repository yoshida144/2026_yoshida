from typing import Callable, List, Dict, Any
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from prox_func import on_l2_sphere_softshrink, on_l2_sphere_relu, group_lasso_in_l2_sphere, group_lasso_on_l2_sphere, no_constraints, in_box_softshrink, on_l2_sphere

torch.backends.cudnn.benchmark = True

# 型定義
DictionaryProx = Callable[
    [torch.Tensor, float],
    torch.Tensor
]

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
      
    def forward(self, y):
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
        
        for t in range(self.T):
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
            
        recon = F.conv_transpose2d(
            x, self.dictionary,
            stride=self.stride,
            padding=self.padding,
            output_padding=(outpadding_h, outpadding_w)
        )
        return x, recon
    
    def forward_eval(self, y, test_data_size, active_threshold=1e-4):
        x, recon = self.forward(y)
        
        l1_norms = torch.norm(x, p=1, dim=(2, 3))
        is_active = (l1_norms > active_threshold).float()
        self.selective_probability += is_active.sum(dim=0) / test_data_size
        
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
            
    def forward(self, images):
        current_input = images
        codes = []
        recons = []
        
        for layer in self.layers:
            code, recon = layer(current_input)
            codes.append(code)
            recons.append(recon)
            current_input = code
        
        final_code = codes[-1]
        return final_code, codes, recons
    
    def forward_eval(self, img, test_data_size):
        current_input = img
        codes = []
        recons = []
        for layer in self.layers:
            code, recon = layer.forward_eval(current_input, test_data_size)
            codes.append(code)
            recons.append(recon)
            current_input = code
            
        final_code = codes[-1]
        return final_code, codes, recons
    
    def decode(self, index, images, codes):
        recon = codes[index]

        for k in range(index, -1, -1):
            layer = self.layers[k]
            
            if k == 0:
                target_h, target_w = images.shape[2], images.shape[3]
            else:
                target_h, target_w = codes[k-1].shape[2], codes[k-1].shape[3]
            
            # Detach control
            if k == index:
                weight = layer.dictionary
            else:
                weight = layer.dictionary
            
            in_h, in_w = recon.shape[2], recon.shape[3]
            stride = layer.stride
            padding = layer.padding
            kernel_size = layer.dictionary.shape[2]
            
            outpadding_h = target_h - ((in_h - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
            outpadding_w = target_w - ((in_w - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
            
            outpadding_h = max(0, outpadding_h)
            outpadding_w = max(0, outpadding_w)
            
            recon = F.conv_transpose2d(
                recon, 
                weight,
                stride=stride,
                padding=padding,
                output_padding=(outpadding_h, outpadding_w)
            )
        return recon
    
    def decode_from_selected_filters(self, layer_idx, images, codes, active_indices):
        original_feature = codes[layer_idx].detach().clone()
        filtered_feature = torch.zeros_like(original_feature)
        
        if isinstance(active_indices, np.ndarray):
            active_indices = active_indices.tolist()
        elif isinstance(active_indices, torch.Tensor):
            active_indices = active_indices.cpu().tolist()
            
        if len(active_indices) > 0:
            filtered_feature[:, active_indices, :, :] = original_feature[:, active_indices, :, :]
        
        curr_recon = filtered_feature
        
        for k in range(layer_idx, -1, -1):
            layer = self.layers[k]
            
            if k == 0:
                target_h, target_w = images.shape[2], images.shape[3]
            else:
                target_h, target_w = codes[k-1].shape[2], codes[k-1].shape[3]
            
            in_h, in_w = curr_recon.shape[2], curr_recon.shape[3]
            stride = layer.stride
            padding = layer.padding
            kernel_size = layer.dictionary.shape[2]
            
            outpadding_h = target_h - ((in_h - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
            outpadding_w = target_w - ((in_w - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
            
            weight = layer.dictionary.detach()
            
            curr_recon = F.conv_transpose2d(
                curr_recon, 
                weight,
                stride=stride,
                padding=padding,
                output_padding=(outpadding_h, outpadding_w)
            )
            
        return curr_recon
    
    def get_effective_dictionary(self, layer_idx):
        curr_dict = self.layers[layer_idx].dictionary.detach()
        
        for k in range(layer_idx - 1, -1, -1):
            layer = self.layers[k]
            
            stride = layer.stride
            
            curr_dict = F.conv_transpose2d(
                curr_dict, 
                layer.dictionary.detach(),
                stride=stride,
                padding=0,
                output_padding=0
            )
            
        return curr_dict
    
class ML_CSC_Image_Classification(nn.Module):
    def __init__(self, input_channels, num_classes, layer_specs, method = "group"):
        super().__init__()
        
        self.feature_extractor = LISTA_ML_CSC(input_channels, num_classes, layer_specs, method = method)
            
        self.final_classifier = nn.Linear(self.feature_extractor.layers[-1].out_channels, num_classes, bias=False)
        #self.final_classifier.weight.data.zero_()
        
    def forward(self, images):
        final_code, codes, recons = self.feature_extractor(images)
        
        vec_features = torch.amax(final_code, dim=(2, 3))
        
        logits = self.final_classifier(vec_features)
        
        return logits, codes, recons
    
    def forward_eval(self, images, test_data_size):
        final_code, codes, recons = self.feature_extractor.forward_eval(images, test_data_size)
        
        vec_features = torch.amax(final_code, dim=(2, 3))
        
        logits = self.final_classifier(vec_features)
        
        return logits, codes, recons
    
    def decode(self, index, images, codes):
        return self.feature_extractor.decode(index, images, codes)
    
    def decode_from_selected_filters(self, layer_idx, images, codes, active_indices):
        return self.feature_extractor.decode_from_selected_filters(layer_idx, images, codes, active_indices)
    
    def decode_single_layer(self, index, code, target_shape):
        layer = self.feature_extractor.layers[index]
        
        target_h, target_w = target_shape[2], target_shape[3]
        
        in_h, in_w = code.shape[2], code.shape[3]
        
        stride = layer.stride
        padding = layer.padding
        kernel_size = layer.dictionary.shape[2]
        
        outpadding_h = target_h - ((in_h - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
        outpadding_w = target_w - ((in_w - 1) * stride - 2 * padding + (kernel_size - 1) + 1)
        
        # 負の値にならないようクリップ（通常は0か1）
        outpadding_h = max(0, outpadding_h)
        outpadding_w = max(0, outpadding_w)
        
        return F.conv_transpose2d(
            code, 
            layer.dictionary, 
            stride=layer.stride, 
            padding=layer.padding,
            output_padding=(outpadding_h, outpadding_w)
        )
    
    def get_effective_dictionary(self, layer_idx):
        return self.feature_extractor.get_effective_dictionary(layer_idx)
    
    @torch.no_grad()
    def prox_classifier(self, threshold):
        self.final_classifier.weight.data = F.softshrink(self.final_classifier.weight.data, threshold)
        
    @torch.no_grad()
    def prox_classifier_group(self, threshold):
        norms = torch.norm(self.final_classifier.weight.data, p=2, dim=0)
        norms = torch.clamp(norms, min=1e-6)
        
        scale = torch.clamp(1 - threshold / norms, min=0)
        
        self.final_classifier.weight.data *= scale.unsqueeze(0)

    def get_pruned_model(self, layer_idx, filter_indices):
        new_model = copy.deepcopy(self)
        target_layer = new_model.feature_extractor.layers[layer_idx]
        
        total_filters = target_layer.out_channels
        
        if isinstance(filter_indices, int): filter_indices = [filter_indices]
        if isinstance(filter_indices, np.ndarray): filter_indices = filter_indices.tolist()
        
        remove_set = set(filter_indices)
        keep_indices = [i for i in range(total_filters) if i not in remove_set]
        keep_indices.sort()
        
        if not keep_indices:
            raise ValueError("Cannot prune all filters!")
            
        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
        
        print(f"Pruning Layer {layer_idx}: Keeping {len(keep_indices)}/{total_filters} filters.")

        new_dict = target_layer.dictionary.data[keep_indices_tensor]
        target_layer.dictionary = nn.Parameter(new_dict)
        target_layer.out_channels = len(keep_indices)
        
        if hasattr(target_layer, 'selective_probability'):
            new_prob = target_layer.selective_probability[keep_indices_tensor]
            target_layer.register_buffer('selective_probability', new_prob)

        is_last_layer = (layer_idx == len(new_model.feature_extractor.layers) - 1)
        
        if is_last_layer:
            old_weight = new_model.final_classifier.weight.data
            new_weight = old_weight[:, keep_indices_tensor]
            
            out_features = new_model.final_classifier.out_features
            new_model.final_classifier = nn.Linear(len(keep_indices), out_features, bias=False)
            new_model.final_classifier.weight = nn.Parameter(new_weight)
            
        else:
            next_layer = new_model.feature_extractor.layers[layer_idx + 1]
            old_next_dict = next_layer.dictionary.data
            new_next_dict = old_next_dict[:, keep_indices_tensor, :, :]
            
            next_layer.dictionary = nn.Parameter(new_next_dict)
            next_layer.in_channels = len(keep_indices)
            
        return new_model
    
    
    def get_effective_dictionary(self, layer_idx):
        return self.feature_extractor.get_effective_dictionary(layer_idx)
    

class ML_CSC_Reconstruction(nn.Module):
    """
    CSR (Convolutional Sparse Representations) 用のモデル。
    分類器を持たず、画像のスパース符号化と再構成のみを行う。
    """
    def __init__(self, input_channels, layer_specs, method="group"):
        super().__init__()
        self.feature_extractor = LISTA_ML_CSC(input_channels, 0, layer_specs, method=method)
        
    def forward(self, images):
        return self.feature_extractor(images)
    
    def forward_eval(self, images, test_data_size):
        return self.feature_extractor.forward_eval(images, test_data_size)
    
    def decode(self, index, images, codes):
        return self.feature_extractor.decode(index, images, codes)
    
    def get_effective_dictionary(self, layer_idx):
        return self.feature_extractor.get_effective_dictionary(layer_idx) 
    
    def decode_single_layer(self, index, code, target_shape):
        return self.feature_extractor.decode_single_layer(index, code, target_shape)
    
    def get_pruned_model(self, layer_idx, filter_indices):
        new_model = copy.deepcopy(self)
        target_layer = new_model.feature_extractor.layers[layer_idx]
        
        total_filters = target_layer.out_channels
        if isinstance(filter_indices, int): filter_indices = [filter_indices]
        if isinstance(filter_indices, np.ndarray): filter_indices = filter_indices.tolist()
        
        remove_set = set(filter_indices)
        keep_indices = [i for i in range(total_filters) if i not in remove_set]
        keep_indices.sort()
        
        if not keep_indices: raise ValueError("Cannot prune all filters!")
        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
        
        new_dict = target_layer.dictionary.data[:, keep_indices_tensor, :, :]
        target_layer.dictionary = nn.Parameter(new_dict)
        target_layer.out_channels = len(keep_indices)
        
        if hasattr(target_layer, 'selective_probability'):
            new_prob = target_layer.selective_probability[keep_indices_tensor]
            target_layer.register_buffer('selective_probability', new_prob)
        
        if layer_idx < len(new_model.feature_extractor.layers) - 1:
            next_layer = new_model.feature_extractor.layers[layer_idx + 1]
            old_next_dict = next_layer.dictionary.data
            new_next_dict = old_next_dict[keep_indices_tensor, :, :, :]
            next_layer.dictionary = nn.Parameter(new_next_dict)
            next_layer.in_channels = len(keep_indices)
            
        return new_model