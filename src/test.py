import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --------------------------------------------------------
# Path Setup & Imports
# --------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import ML_CSC_Image_Classification
from utils import load_config, load_dataset, load_dataset_info

# ==========================================
# 1. Setup & Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_TEST_PATH = "./config_test.json"
test_params = load_config(CONFIG_TEST_PATH)

# --- Update these paths as needed ---
TEST_PATH = f"./models/{test_params['test_path']}" 
MODEL_PATH = f"{TEST_PATH}/model.pth"
RESULT_DIR = f"{TEST_PATH}/result"
CONFIG_PATH = f"{TEST_PATH}/config.json"
os.makedirs(RESULT_DIR, exist_ok=True)

params = load_config(CONFIG_PATH)
TARGET_DATASET = params["target_dataset"]
BATCH_SIZE = params["batch_size"]

LAYERS_SPECS = [
    {'dim': s["dim"], 'kernel': s["kernel"], 'stride': s["stride"], 'T': s["T"], 'prox': s["prox"]} 
    for s in params["layer_specs"]
]

dataset_info = load_dataset_info(TARGET_DATASET)
INPUT_CHANNELS = dataset_info["input_channels"]
IMG_SIZE = dataset_info["img_size"]
NUM_CLASSES = dataset_info["num_classes"]
DATA_MEAN = dataset_info["data_mean"]
DATA_STD = dataset_info["data_std"]
CLASSES = dataset_info["classes"]

VIS_CMAP = 'gray' if INPUT_CHANNELS == 1 else None

print(f"Dataset: {TARGET_DATASET}, Device: {DEVICE}")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DATA_MEAN, DATA_STD),
])

_, dataset = load_dataset(TARGET_DATASET, None, test_transform)

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

model = ML_CSC_Image_Classification(
    input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES, layer_specs=LAYERS_SPECS
).to(DEVICE)

print(f"Loading weights from {MODEL_PATH}...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.eval()


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
    
    activations = torch.amax(final_code, dim=(2, 3)).squeeze() 
    
    classifier = model.classifier if hasattr(model, 'classifier') else model.final_classifier
    W = classifier.weight.data 
    num_classes = W.shape[0]
    
    w_target = W[label_idx]
    
    all_indices = torch.arange(num_classes, device=W.device)
    other_indices = all_indices[all_indices != label_idx]
    
    w_others = W[other_indices]
    
    w_others_sum = torch.sum(w_others, dim=0)
    total_margin_weights = (len(other_indices) * w_target) - w_others_sum
    
    contribution = (activations * total_margin_weights).cpu().numpy()
    
    sorted_indices = np.argsort(contribution)[::-1]
    
    return contribution, sorted_indices, activations.cpu().numpy(), total_margin_weights.cpu().numpy()

def experiment_incremental_breakdown(model, img, codes, sorted_indices, idx, predicted_class, top_k=10):
    """
    [New Experiment] Incremental Breakdown Visualization (Enhanced Reconstruction Visibility)
    
    Displays a grid for the top-k filters:
    1. Effective Dictionary Filter (filter)
    2. Sparse Coefficient Map (Activation) -> Original Raw Map
    3. Single Reconstruction (Contribution of this filter alone) -> ★Normalized for Visibility (Exp2 style)
    4. Cumulative Reconstruction (Sum of filters up to this rank) -> ★Normalized for Visibility (Exp2 style)
    """
    print(f"   -> Running Incremental Breakdown (Top-{top_k}) [High Contrast Recons]...")
    
    last_layer_idx = len(model.feature_extractor.layers) - 1
    
    eff_dicts = model.get_effective_dictionary(last_layer_idx).detach().cpu()
    
    fig, axes = plt.subplots(top_k, 4, figsize=(16, 3.0 * top_k))
    
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
        im_code = axes[rank, 1].imshow(code_map, cmap='magma', vmin=0, vmax=vmax)
        
        plt.colorbar(im_code, ax=axes[rank, 1], fraction=0.046, pad=0.02)
        
        # --- Col 3: Single Reconstruction ---
        recon_single = model.decode_from_selected_filters(last_layer_idx, img, codes, [filter_idx])
        axes[rank, 2].imshow(normalize_vis_image(recon_single), cmap=VIS_CMAP)
        
        # --- Col 4: Cumulative Reconstruction ---
        recon_cum = model.decode_from_selected_filters(last_layer_idx, img, codes, current_indices)
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
    top_n = 5
    targets = sorted_indices[:top_n]
    
    fig, axes = plt.subplots(1, top_n + 1, figsize=(18, 5))
    plt.suptitle(f"Exp 2: Top Score filters (Sample {idx})", fontsize=16)
    
    axes[0].imshow(to_cpu_img(img), cmap=VIS_CMAP)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, filter_idx in enumerate(targets):
        # 1. Reconstruction
        recon_single = model.decode_from_selected_filters(last_layer_idx, img, codes, [filter_idx])
        
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


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 描画テスト")
    print("="*50)

    analyze_dictionaries(model)

    target_indices = [0, 1, 2] 
    print(f"\n再構成画像: {target_indices}")
    
    for idx in target_indices:
        print(f" 実行中({idx})...")
        img, label = dataset[idx]
        img_tensor = img.unsqueeze(0).to(DEVICE)
        analyze_reconstruction(model, img_tensor, idx)
        
        with torch.no_grad():
            logits, codes, _ = model(img_tensor)
            
        pred_idx = logits.argmax(dim=1).item()
        pred_class_name = CLASSES[pred_idx]
        true_class_name = CLASSES[label]
        
        print(f"予測: {pred_class_name} (正解: {true_class_name})")
        
        contribution, sorted_indices, activations, score_weights = get_filter_contributions(model, codes, logits, pred_idx)
        
        experiment_single_filter_roles(model, img_tensor, codes, sorted_indices, activations, score_weights, idx, f"{pred_class_name}(True:{true_class_name})")
        
        experiment_incremental_breakdown(model, img_tensor, codes, sorted_indices, idx, pred_class_name, top_k=5)
        
    
    scan_range = range(len(dataset)) 
    
    print(f"Scanning for misclassified samples in range: 0 - {len(dataset)}...")
    
    count = 0
    max_samples = 10 # 解析したい誤分類サンプルの最大数
    
    for idx in scan_range:
        if count >= max_samples:
            break
            
        img, label = dataset[idx]
        img_tensor = img.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits, codes, _ = model(img_tensor)
            
        pred_idx = logits.argmax(dim=1).item()
        pred_class_name = CLASSES[pred_idx]
        true_class_name = CLASSES[label]
        
        # 予測が間違っている場合のみ実行
        if pred_idx != label:
            count += 1
            print(f"\n[誤サンプル数 #{count}] Sample {idx}")
            print(f"予測: {pred_class_name} (正解: {true_class_name})")
            
            contribution, sorted_indices, activations, score_weights = get_filter_contributions(model, codes, logits, pred_idx)
            
            # 単一フィルターの分類性能
            experiment_single_filter_roles(model, img_tensor, codes, sorted_indices, activations, score_weights, idx, f"{pred_class_name}(True:{true_class_name})")
            
            # 誤った理由の可視化
            experiment_incremental_breakdown(model, img_tensor, codes, sorted_indices, idx, f"{pred_class_name}(True:{true_class_name})", top_k=5)
        
    print("\n" + "="*50)
    print(f"保存先: {RESULT_DIR}")
    print("="*50)