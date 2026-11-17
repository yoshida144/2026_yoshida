import sys
import os
import json
import argparse
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# Path Setup & Imports
# --------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import ML_CSC_Image_Classification, ML_CSC_Reconstruction
from utils import load_config, build_loaders, set_execution

# Initialized only by main(): importing this module never loads data or weights.
DEVICE = torch.device("cpu")
RESULT_DIR = None
DATA_MEAN = None
DATA_STD = None
VIS_CMAP = None


def selected_reconstruction(model, layer_idx, images, codes, indices):
    # Both classifier and reconstruction models share this decoder.
    return model.feature_extractor.decode_from_selected_filters(
        layer_idx, images, codes, indices
    )


def to_cpu_img(t):
    t = t.detach().cpu().clone().squeeze(0)
    for c in range(t.shape[0]):
        t[c] = t[c] * DATA_STD[c] + DATA_MEAN[c]
    t = t.permute(1, 2, 0).clamp(0, 1).numpy() 
    if t.shape[2] == 1: t = t.squeeze(2)
    return t

def normalize_vis(filter_weight):
    limit = np.max(np.abs(filter_weight))
    if limit < 1e-8: limit = 1.0
    img = filter_weight / limit
    img = 0.5 * img + 0.5
    return np.clip(img, 0, 1)

def normalize_vis_image(t):
    img = t.detach().cpu().clone()
    if img.dim() == 4: img = img.squeeze(0)
    img = img.numpy()
    limit = np.max(np.abs(img))
    if limit < 1e-8: limit = 1.0
    img = img / limit
    img = img * 0.5  + 0.5
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    if img.shape[2] == 1: img = img.squeeze(2)
    return img


def analyze_dictionaries(model):
    print("\n有効辞書の描画")
    
    layers = model.feature_extractor.layers if hasattr(model, 'feature_extractor') else model.layers
    
    input_channels = layers[0].dictionary.shape[1] 
    cmap = 'gray' if input_channels == 1 else None
    
    for l_idx in range(len(layers)):
        d_eff = model.get_effective_dictionary(l_idx).detach().cpu().numpy()
        
        h_eff, w_eff = d_eff.shape[2], d_eff.shape[3]
        num_filters = d_eff.shape[0]
        num_show = num_filters
        
        grid_size = int(np.ceil(np.sqrt(num_show)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        plt.suptitle(f"stage {l_idx+1} effective dictionaries (size: {h_eff}x{w_eff})")
        
        if grid_size == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for k in range(len(axes)):
            if k < num_show:
                filter = d_eff[k].transpose(1, 2, 0)
                filter = normalize_vis(filter)
                
                if filter.shape[2] == 1: 
                    axes[k].imshow(filter.squeeze(2), cmap=cmap)
                else:
                    axes[k].imshow(filter)
                
            axes[k].axis('off')
            
        save_path = f"{RESULT_DIR}/dictionary_layer{l_idx+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f" Saved Layer {l_idx+1}: {save_path} (Size: {h_eff}x{w_eff})")

def analyze_reconstruction(model, img, idx):
    with torch.no_grad(): 
        _, codes, _ = model(img.to(DEVICE))
    
    fig, axes = plt.subplots(1, len(codes) + 1, figsize=(15, 5))
    
    axes[0].imshow(to_cpu_img(img), cmap=VIS_CMAP)
    axes[0].set_title("original")
    axes[0].axis('off')
    
    for i in range(len(codes)):
        decoded = model.decode(i, img.to(DEVICE), codes)
        
        axes[i+1].imshow(to_cpu_img(decoded), cmap=VIS_CMAP)
        axes[i+1].set_title(f"Stage {i+1} Recon")
        axes[i+1].axis('off')
        
    save_path = f"{RESULT_DIR}/sample{idx}_reconstruction.png"
    plt.savefig(save_path)
    plt.close()
    print(f" Saved Reconstruction: {save_path}")
    
def get_filter_contributions(model, codes, logits, label_idx):
    final_code = codes[-1] 
    
    activations = torch.amax(final_code, dim=(2, 3)).squeeze(0) 
    
    classifier = model.classifier if hasattr(model, 'classifier') else model.final_classifier
    W = classifier.weight.data 
    num_classes = W.shape[0]
    
    w_target = W[label_idx]
    
    all_indices = torch.arange(num_classes, device=W.device)
    other_indices = all_indices[all_indices != label_idx]
    
    w_others = W[other_indices]
    
    w_others_sum = torch.sum(w_others, dim=0)
    total_margin_weights = (len(other_indices) * w_target) - w_others_sum
    
    contribution = (activations * total_margin_weights).detach().cpu().numpy()
    
    sorted_indices = np.argsort(contribution)[::-1]
    
    return contribution, sorted_indices, activations.detach().cpu().numpy(), total_margin_weights.detach().cpu().numpy()

def experiment_incremental_breakdown(model, img, codes, sorted_indices, idx, predicted_class, top_k=10):
    """
    [New Experiment] Incremental Breakdown Visualization (Enhanced Reconstruction Visibility)
    
    Displays a grid for the top-k filters:
    1. Effective Dictionary Filter (filter)
    2. Sparse Coefficient Map (Activation) -> Original Raw Map
    3. Single Reconstruction (Contribution of this filter alone) -> ★Normalized for Visibility (Exp2 style)
    4. Cumulative Reconstruction (Sum of filters up to this rank) -> ★Normalized for Visibility (Exp2 style)
    """
    top_k = min(top_k, len(sorted_indices))
    if top_k < 1:
        return
    print(f"   -> Running Incremental Breakdown (Top-{top_k}) [High Contrast Recons]...")
    
    last_layer_idx = len(model.feature_extractor.layers) - 1
    
    eff_dicts = model.get_effective_dictionary(last_layer_idx).detach().cpu()
    
    fig, axes = plt.subplots(top_k, 4, figsize=(16, 3.0 * top_k), squeeze=False)
    
    cols = ["Filter filter (Shape)", "Sparse Code (Activation)", "Single Recon (D*z)", "Cumulative Recon"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=10)

    current_indices = []
    
    for rank in range(top_k):
        filter_idx = sorted_indices[rank]
        current_indices.append(filter_idx)
        
        # --- Col 1: Dictionary Filter (filter) ---
        filter = eff_dicts[filter_idx] 
        vis_filter = normalize_vis_image(filter)
        
        axes[rank, 0].imshow(vis_filter, cmap=VIS_CMAP)
        axes[rank, 0].set_ylabel(f"Rank #{rank+1}\nFilter {filter_idx}", fontsize=10, fontweight='bold', rotation=90, labelpad=10)
        
        # --- Col 2: Sparse Code Map (Original) ---
        code_map = codes[-1][0, filter_idx].detach().cpu().numpy()
        vmax = np.max(np.abs(code_map))
        if vmax < 1e-8: vmax = 1.0
        im_code = axes[rank, 1].imshow(code_map, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        
        plt.colorbar(im_code, ax=axes[rank, 1], fraction=0.046, pad=0.02)
        
        # --- Col 3: Single Reconstruction ---
        recon_single = selected_reconstruction(model, last_layer_idx, img, codes, [filter_idx])
        axes[rank, 2].imshow(normalize_vis_image(recon_single), cmap=VIS_CMAP)
        
        # --- Col 4: Cumulative Reconstruction ---
        recon_cum = selected_reconstruction(model, last_layer_idx, img, codes, current_indices)
        axes[rank, 3].imshow(normalize_vis_image(recon_cum), cmap=VIS_CMAP)
        
        for i, ax in enumerate(axes[rank]):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.axis('on') 
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
            else:
                 pass

    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.05, right=0.95, top=0.92, bottom=0.02)
    
    save_path = f"{RESULT_DIR}/sample{idx}_exp_incremental_breakdown.png"
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close()
    print(f"     Saved: {save_path}")
    

def experiment_single_filter_roles(model, img, codes, sorted_indices, activations, score_weights, idx, predicted_class):
    """
    Visualizes single filters based on Score ranking.
    Displays Activation (z) and Margin Weight (w_diff) in the title.
    """
    last_layer_idx = len(model.feature_extractor.layers) - 1
    top_n = min(5, len(sorted_indices))
    targets = sorted_indices[:top_n]
    
    fig, axes = plt.subplots(1, top_n + 1, figsize=(18, 5))
    plt.suptitle(f"Exp 2: Top Score filters (Sample {idx})", fontsize=16)
    
    axes[0].imshow(to_cpu_img(img), cmap=VIS_CMAP)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, filter_idx in enumerate(targets):
        # 1. Reconstruction
        recon_single = selected_reconstruction(model, last_layer_idx, img, codes, [filter_idx])
        
        # 2. Metrics
        z_val = activations[filter_idx]
        w_margin_val = score_weights[filter_idx]
        score_val = z_val * w_margin_val
        
        # 3. Display
        axes[i+1].imshow(normalize_vis_image(recon_single), cmap=VIS_CMAP)
        
        title_str = (f"Rank #{i+1} (Filter {filter_idx})\n"
                     f"Score: {score_val:.2f}\n"
                     f"z={z_val:.2f}, margin={w_margin_val:.2f}")
        
        axes[i+1].set_title(title_str, fontsize=10)
        axes[i+1].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{RESULT_DIR}/sample{idx}_exp2_single_filters.png")
    plt.close()


def main(argv=None):
    global DEVICE, RESULT_DIR, DATA_MEAN, DATA_STD, VIS_CMAP
    parser = argparse.ArgumentParser(description="Original visual analyses, adapted for LISTA/LIF and CSR/classification")
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', choices=['auto', 'csr', 'classification'], default='auto')
    parser.add_argument('--mode', choices=['dense', 'event'], default='dense')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-misclassified', type=int, default=10)
    parser.add_argument('--scan-limit', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 0 or args.top_k < 1 or args.max_misclassified < 0:
        parser.error('samples/max-misclassified must be >=0 and top-k >=1')
    if args.scan_limit is not None and args.scan_limit < 1:
        parser.error('--scan-limit must be >=1')

    model_dir = args.model_dir
    if model_dir is None and args.checkpoint is not None:
        model_dir = args.checkpoint.parent
    if model_dir is None:
        legacy_path = Path('config_test.json')
        if not legacy_path.exists():
            parser.error('Specify --model-dir or --checkpoint, or provide config_test.json')
        legacy = load_config(str(legacy_path))
        model_dir = Path('models') / legacy['test_path']
    config_path = args.config or model_dir / 'config.json'
    params = load_config(str(config_path))

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted(model_dir.glob('*.pth'))
        if len(candidates) != 1:
            parser.error('Specify --checkpoint when the directory contains zero or multiple .pth files')
        checkpoint = candidates[0]
    DEVICE = torch.device(args.device)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint must contain a nonempty state_dict')

    is_lif = any(key.endswith('.Wy') for key in state)
    is_classifier = 'final_classifier.weight' in state
    detected_task = 'classification' if is_classifier else 'csr'
    if args.task != 'auto' and args.task != detected_task:
        raise ValueError(f'Requested {args.task}, checkpoint is {detected_task}')
    specs = copy.deepcopy(params['layer_specs'])
    for index, spec in enumerate(specs):
        layer_is_lif = f'feature_extractor.layers.{index}.Wy' in state
        detected_mode = ('lif_budget' if f'feature_extractor.layers.{index}.input_gain' in state
                         else 'lif' if layer_is_lif else 'lista')
        if 'model' in spec and spec['model'] != detected_mode:
            raise ValueError(f'Layer {index}: config model and checkpoint disagree')
        spec.setdefault('model', detected_mode)
        # All extra LIF options and T/stride/prox are retained from the saved config.
    method = params.get('method', 'group' if any('.lams_group' in k for k in state) else 'none')
    dataset_params = copy.deepcopy(params)
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    dataset_params['batch_size'] = args.batch_size
    if args.synthetic:
        dataset_params['synthetic'] = True
    print('Loading evaluation dataset...', flush=True)
    _, test_loader, info = build_loaders(dataset_params, evaluation_only=True)
    dataset = test_loader.dataset
    DATA_MEAN, DATA_STD = info['data_mean'], info['data_std']
    classes = info['classes']
    VIS_CMAP = 'gray' if info['input_channels'] == 1 else None

    if is_classifier:
        model = ML_CSC_Image_Classification(info['input_channels'], info['num_classes'], specs, method)
    else:
        model = ML_CSC_Reconstruction(info['input_channels'], specs, method)
    model = model.to(DEVICE)
    # Never hide missing Wy/Wz/a or frozen initial dictionaries.
    model.load_state_dict(state, strict=True)
    model.eval()
    set_execution(model, args.mode)
    RESULT_DIR = str(args.output or model_dir / ('result_lif_' + args.mode if is_lif else 'result_lista'))
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    correct = samples = elements = 0
    squared_error = raw_squared_error = 0.0
    spikes = opportunities = 0
    misclassified = []
    torch.manual_seed(43)
    with torch.no_grad():
        for images, labels in test_loader:
            if args.scan_limit is not None:
                remaining = args.scan_limit - samples
                if remaining <= 0:
                    break
                images, labels = images[:remaining], labels[:remaining]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            print(f'Evaluating {args.mode}: images {samples+1}-{samples+len(images)}', flush=True)
            result, codes, _ = model(images)
            reconstruction = model.decode(len(codes)-1, images, codes)
            error = reconstruction-images
            squared_error += error.square().sum().item()
            std = images.new_tensor(DATA_STD).view(1,-1,1,1)
            raw_squared_error += (error*std).square().sum().item()
            elements += images.numel()
            if is_classifier:
                predictions = result.argmax(1)
                correct += int((predictions == labels).sum())
                for offset in (predictions != labels).nonzero(as_tuple=False).flatten().tolist():
                    if len(misclassified) < args.max_misclassified:
                        misclassified.append(samples + offset)
            for layer in model.feature_extractor.layers:
                stats = getattr(layer, 'last_stats', {})
                spikes += stats.get('spikes', 0)
                opportunities += stats.get('opportunities', 0)
            samples += len(images)
        if samples == 0:
            raise ValueError('Empty evaluation dataset')

        analyze_dictionaries(model)
        target_indices = list(range(min(args.samples, len(dataset))))
        for idx in dict.fromkeys(target_indices + misclassified):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(DEVICE)
            result, codes, _ = model(image)
            analyze_reconstruction(model, image, idx)
            if is_classifier:
                predicted = result.argmax(1).item()
                _, indices, activations, weights = get_filter_contributions(model, codes, result, predicted)
                title = f'{classes[predicted]} (True:{classes[int(label)]})'
                experiment_single_filter_roles(model, image, codes, indices, activations, weights, idx, title)
            else:
                # No classifier exists in CSR; rank by code-map L2 energy instead.
                energies = codes[-1][0].flatten(1).norm(p=2, dim=1)
                indices = energies.argsort(descending=True).cpu().numpy()
                title = 'CSR code-energy ranking'
            experiment_incremental_breakdown(model, image, codes, indices, idx, title, top_k=args.top_k)

    raw_mse = raw_squared_error/elements
    summary = {
        'checkpoint': str(checkpoint), 'config': str(config_path),
        'task': detected_task, 'lif': is_lif, 'mode': args.mode,
        'synthetic': dataset_params.get('synthetic', False),
        'samples': samples, 'mse_normalized': squared_error/elements,
        'psnr_db_unclipped_unit_range': float(-10*np.log10(max(raw_mse,1e-30))),
        'accuracy': correct/samples if is_classifier else None,
        'spike_rate': spikes/opportunities if opportunities else None,
        'misclassified_visualized': misclassified,
        'notes': ['No noise is generated. Input and reconstruction target are the loaded image.',
                  'Filter panels are individually contrast-normalized visualizations, not numerical accuracy evidence.',
                  'Classification rankings retain the original global-max / margin-weight score.',
                  'CSR rankings use code L2 energy, not classifier contribution.']
    }
    Path(RESULT_DIR, 'evaluation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved analyses to: {RESULT_DIR}')
    return summary


if __name__ == '__main__':
    main()
