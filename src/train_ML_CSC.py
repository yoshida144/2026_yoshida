import sys
import os
import datetime
import json
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import ML_CSC_Reconstruction 
from utils import load_config, load_dataset, load_dataset_info

def train_csr(device, params, model_path=None, save_model_name=None):
    
    DEVICE = device

    TARGET_DATASET = params["target_dataset"]
    BATCH_SIZE = params["batch_size"]
    NUM_EPOCHS = params["num_epochs"]
    LR = params["learning_rate"]
    LR_LAMS = params["learning_rate_lambdas"]
    
    # Loss Weights
    # LAMBDA_RECON = params["lambda_recon"] 主目的なのでこの係数は1で固定
    LAMBDA_SPARSE = params["lambda_sparse"]
    LAMBDA_DICT = params["lambda_dict"]
    LAMBDA_GROUP = params["lambda_group"]
    # LAMBDA_CLASS = params["lambda_class"] 分類器が無いので不要
    
    METHOD = params["method"]

    LAYERS_SPECS = [
        {
            'dim': spec["dim"],
            'kernel': spec["kernel"],
            'stride': spec["stride"],
            'T': spec["T"],
            'prox': spec["prox"]
        } for spec in params["layer_specs"]
    ]

    dataset_info = load_dataset_info(TARGET_DATASET)
    INPUT_CHANNELS = dataset_info["input_channels"]
    IMG_SIZE = dataset_info["img_size"]
    DATA_MEAN = dataset_info["data_mean"]
    DATA_STD = dataset_info["data_std"]

    # ============================================================
    # Dataset
    # ============================================================
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])

    train_dataset, test_dataset = load_dataset(
        TARGET_DATASET, train_transform, test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # ============================================================
    # Model Initialization (CSR Model)
    # ============================================================
    print(f"Initializing CSR Model for {TARGET_DATASET}...")
    model = ML_CSC_Reconstruction(
        input_channels=INPUT_CHANNELS,
        layer_specs=LAYERS_SPECS,
        method=METHOD
    ).to(DEVICE)

    if model_path is not None:
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        else:
            print(f"Warning: Model path {model_path} not found. Starting from scratch.")

    # ============================================================
    # Optimizer
    # ============================================================
    param_lams = []
    param_dicts = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lams' in name or 'etas' in name:
            param_lams.append(param)
        else:
            # dictionary param
            param_dicts.append(param)

    optimizer = optim.SGD([
        {'params': param_lams, 'lr': LR_LAMS},
        {'params': param_dicts, 'lr': LR}
    ])

    # ============================================================
    # Logging
    # ============================================================
    log = {
        "epoch": [],
        "train_mse": [],
        "test_mse": [],
        "loss_total": [],
        "loss_recon": [],
        "loss_sparse": [],
    }

    # ============================================================
    # Training Loop
    # ============================================================
    print(f"Start CSR Training (Reconstruction & Sparsity)...")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        model.train()

        epoch_loss_r = 0.0
        epoch_loss_s = 0.0
        epoch_loss_total = 0.0
        epoch_mse_accum = 0.0

        train_bar = tqdm(train_loader, desc="Train", leave=False)

        for images, _ in train_bar: # Labels are ignored
            images = images.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Forward: Get codes and intermediate reconstructions
            final_code, codes, recons = model(images)

            # --- 1. Reconstruction Loss ---
            loss_r = 0.0
            
            # 各層の再構成誤差を合計 (Multi-layer Reconstruction Loss)
            # D_l * z_l が Input に近づくようにする
            for l in range(len(codes)):
                # model.decode を使用 (ConvTranspose2d + OutputPadding)
                recon_l = model.decode(index=l, images=images, codes=codes)
                
                # 画素数で正規化しないとLossが大きくなりすぎるので調整しても良い
                # ここでは単純和をとる
                loss_r += F.mse_loss(recon_l, images) 

            # --- 2. Sparse Penalty ---
            loss_s = 0.0
            loss_g = 0.0
            
            for code in codes:
                # L1 Norm
                l1_term = torch.mean(torch.norm(code, p=1, dim=(1, 2, 3)))
                loss_s += l1_term
                
                # Group sparse
                if LAMBDA_GROUP > 0:
                    # L2 norm over spatial dimensions (H, W), then sum over channels
                    group_energies = torch.norm(code, p=2, dim=(2, 3)) 
                    group_term = torch.mean(torch.sum(group_energies, dim=1))
                    loss_g += group_term

            # Total Loss
            total_loss = (
                loss_r
                + LAMBDA_SPARSE * loss_s
                + LAMBDA_GROUP * loss_g
            )
            
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # --- Proximal Operator (Project Dictionaries) ---
            # model.feature_extractor.layers を走査
            for i, layer in enumerate(model.feature_extractor.layers):
                layer.prox_function(LAMBDA_DICT)

            # Logging
            epoch_loss_r += loss_r.item()
            epoch_loss_s += loss_s.item()
            epoch_loss_total += total_loss.item()
            
            # Monitor Final Layer Reconstruction Quality (MSE)
            with torch.no_grad():
                final_recon = model.decode(len(codes)-1, images, codes)
                epoch_mse_accum += F.mse_loss(final_recon, images).item()

            train_bar.set_postfix({
                "L": f"{total_loss.item():.2e}",
                "Rec": f"{loss_r.item():.1e}",
                "Sp": f"{loss_s.item():.1e}",
            })

        # --- Evaluation ---
        model.eval()
        test_mse_accum = 0.0
        test_bar = tqdm(test_loader, desc="Test", leave=False)

        with torch.no_grad():
            for images, _ in test_bar:
                images = images.to(DEVICE)
                _, codes, _ = model(images)
                
                # 最終層からの再構成MSEを計測
                final_recon = model.decode(len(codes)-1, images, codes)
                test_mse_accum += F.mse_loss(final_recon, images).item()

        avg_train_mse = epoch_mse_accum / len(train_loader)
        avg_test_mse = test_mse_accum / len(test_loader)

        print(
            f"Loss Total: {epoch_loss_total/len(train_loader):.4f} | "
            f"Rec: {epoch_loss_r/len(train_loader):.4f} | "
            f"Sparse: {epoch_loss_s/len(train_loader):.4f}"
        )
        print(f"Train MSE: {avg_train_mse:.5f} | ★ Test MSE: {avg_test_mse:.5f}")

        # Update Logs
        log["epoch"].append(epoch + 1)
        log["train_mse"].append(avg_train_mse)
        log["test_mse"].append(avg_test_mse)
        log["loss_total"].append(epoch_loss_total / len(train_loader))
        log["loss_recon"].append(epoch_loss_r / len(train_loader))
        log["loss_sparse"].append(epoch_loss_s / len(train_loader))

    # ============================================================
    # Save Results
    # ============================================================
    dir_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_CSR')
    save_dir = f"models/{dir_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = save_model_name if save_model_name else "model_csr"
    torch.save(model.state_dict(), f"{save_dir}/{save_name}.pth")

    # Config Save
    model_spec = {
        "target_dataset": TARGET_DATASET,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "lambda_recon": 1.0,
        "lambda_sparse": LAMBDA_SPARSE,
        "lambda_group": LAMBDA_GROUP,
        "method": METHOD,
        "layer_specs": params["layer_specs"],
        "note": "CSR Training (No Classifier)"
    }

    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(model_spec, f, indent=2)

    with open(f"{save_dir}/training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Plot MSE
    plt.figure()
    plt.plot(log["epoch"], log["train_mse"], label="Train MSE")
    plt.plot(log["epoch"], log["test_mse"], label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reconstruction Quality (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/mse_curve.png")
    plt.close()

    # Plot Loss Components
    plt.figure()
    plt.plot(log["epoch"], log["loss_total"], label="Total")
    plt.plot(log["epoch"], log["loss_recon"], label="Recon")
    plt.plot(log["epoch"], log["loss_sparse"], label="Sparse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    print(f"CSR Training Finished. Saved to {save_dir}")
    return save_dir


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    CONFIG_PATH = "./config_ml_csr.json"
    params = load_config(CONFIG_PATH)
    
    train_csr(device=DEVICE, params=params, save_model_name="model_csr")