import sys
import os
import datetime
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import spike_penalty, build_loaders
from model import ML_CSC_Image_Classification
from utils import load_config, load_dataset, load_dataset_info

def train(device, params, model_path=None, save_model_name=None):
    
    DEVICE = device

    TARGET_DATASET = params["target_dataset"]
    BATCH_SIZE = params["batch_size"]
    NUM_EPOCHS = params["num_epochs"]
    LR = params["learning_rate"]
    LR_LAMS = params["learning_rate_lambdas"]
    LAMBDA_RECON = params["lambda_recon"]
    LAMBDA_SPARSE = params["lambda_sparse"]
    LAMBDA_DICT = params["lambda_dict"]
    LAMBDA_GROUP = params["lambda_group"]
    LAMBDA_CLASS = params["lambda_class"]
    
    METHOD = params["method"]

    LAYERS_SPECS = [
        {
            'dim': spec["dim"],
            'kernel': spec["kernel"],
            'stride': spec["stride"],
            'T': spec["T"],
            'model': spec.get('model', 'lista'),
            'lif': spec.get('lif', {}),
            'prox':spec["prox"]
        } for spec in params["layer_specs"]
    ]

    train_loader, test_loader, dataset_info = build_loaders(params, classification=True)
    INPUT_CHANNELS = dataset_info["input_channels"]
    NUM_CLASSES = dataset_info["num_classes"]

    model = ML_CSC_Image_Classification(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        layer_specs=LAYERS_SPECS,
        method=METHOD
    ).to(DEVICE)

    # --- Load Pre-trained Weights if provided ---
    if model_path is not None:
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True), strict=True)
        else:
            print(f"Warning: Model path {model_path} not found. Starting from scratch.")

    # ============================================================
    # Optimizer
    # ============================================================
    param_lams = []
    param_other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lams' in name:
            param_lams.append(param)
        else:
            param_other.append(param)

    optimizer = optim.Adam([
        {'params': param_lams, 'lr': LR_LAMS},
        {'params': param_other, 'lr': LR}
    ])

    criterion_cls = nn.CrossEntropyLoss()

    # ============================================================
    # Logging buffers
    # ============================================================
    log = {
        "epoch": [],
        "train_acc": [],
        "test_acc": [],
        "loss_total": [],
        "loss_cls": [],
        "loss_recon": [],
        "loss_sparse": [],
    }

    # ============================================================
    # Training Loop
    # ============================================================
    print("Start Training...")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        model.train()

        epoch_loss_c = 0.0
        epoch_loss_r = 0.0
        epoch_loss_s = 0.0
        epoch_loss_total = 0.0

        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="Train", leave=False)

        for images, labels in train_bar:
            # 1. まずクリーンな画像をデバイスに転送 (これが正解データ: Target)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                logits, codes, _ = model(images)

                loss_c = criterion_cls(logits, labels)

                loss_r = 0.0
                N = images.numel() / images.shape[0]
                
                layers_obj = model.feature_extractor.layers if hasattr(model, 'feature_extractor') else model.layers
                
                for l in ([len(codes)-1] if params.get('reconstruction_loss') == 'final' else range(len(layers_obj))):
                    recon = model.decode(index=l, images=images, codes=codes)
                    
                    loss_r += F.mse_loss(recon, images) * N

                # ====================================================
                # Sparse Penalty (L1 + Group Lasso)
                # ====================================================
                loss_s = 0.0
                loss_g = 0.0
                for code in codes:
                    # 1. L1 Norm (Pixel-wise sparsity)
                    # これまで通り。画素ごとのスパース性を要求。
                    l1_term = torch.mean(torch.norm(code, p=1, dim=(1, 2, 3)))
                    
                    # 2. Group Lasso (Channel-wise L2,1 Norm)
                    # チャンネルごとのエネルギー (B, C) を計算し、その総和をとる
                    # (B, C, H, W) -> (B, C) -> mean over B, sum over C
                    # ※ dim=(2,3) で空間方向のノルム(L2)をとり、そのあと平均をとる
                    group_energies = torch.norm(code, p=2, dim=(2, 3))
                    group_term = torch.mean(torch.sum(group_energies, dim=1))
                    
                    loss_s += l1_term
                    loss_g += group_term

                total_loss = (
                    loss_c
                    + LAMBDA_RECON * loss_r
                    + LAMBDA_SPARSE * loss_s
                    + LAMBDA_GROUP * loss_g
                )
                
            total_loss = total_loss + params.get('lambda_spike', 0.0) * spike_penalty(model)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Proximal Operator
            for i, layer in enumerate(layers_obj):
                layer.prox_function(
                    LAMBDA_DICT 
                )
            model.prox_classifier(threshold=LAMBDA_CLASS)

            epoch_loss_c += loss_c.item()
            epoch_loss_r += loss_r.item()
            epoch_loss_s += loss_s.item()
            epoch_loss_total += total_loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix({
                "L": f"{total_loss.item():.2e}",
                "cls": f"{loss_c.item():.1e}",
                "rec": f"{loss_r.item():.1e}",
                "ℓ1": f"{loss_s.item():.1e}",
            })

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        test_bar = tqdm(test_loader, desc="Test", leave=False)

        with torch.no_grad():
            for images, labels in test_bar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits, _, _ = model(images)
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        test_acc = 100.0 * test_correct / test_total

        print(
            f"Loss: {epoch_loss_total/len(train_loader):.4f} "
            f"(Cls {epoch_loss_c/len(train_loader):.4f}, "
            f"Rec {epoch_loss_r/len(train_loader):.4f}, "
            f"Sparse {epoch_loss_s/len(train_loader):.4f})"
        )
        print(f"Train Acc: {train_acc:.2f}% | ★ Test Acc: {test_acc:.2f}%")

        # ---- log ----
        log["epoch"].append(epoch + 1)
        log["train_acc"].append(train_acc)
        log["test_acc"].append(test_acc)
        log["loss_total"].append(epoch_loss_total / len(train_loader))
        log["loss_cls"].append(epoch_loss_c / len(train_loader))
        log["loss_recon"].append(epoch_loss_r / len(train_loader))
        log["loss_sparse"].append(epoch_loss_s / len(train_loader))


    # ============================================================
    # Save results
    # ============================================================
    
    dir_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = f"models/{dir_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_name = save_model_name or "model"
    torch.save(model.state_dict(), f"{save_dir}/{checkpoint_name}.pth")

    model_spec = {
        "target_dataset": TARGET_DATASET,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "learning_rate_lambdas": LR_LAMS,
        "lambda_recon": LAMBDA_RECON,
        "lambda_sparse": LAMBDA_SPARSE,
        "lambda_dict": LAMBDA_DICT,
        "lambda_group": LAMBDA_GROUP,
        "lambda_class": LAMBDA_CLASS,
        "method":METHOD,
        "layer_specs": LAYERS_SPECS,
        "base_model": model_path if model_path else "None"
    }

    with open(f"{save_dir}/config.json", "w") as f:
        json.dump({**params, **model_spec}, f, indent=2)

    with open(f"{save_dir}/training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ============================================================
    # Plot
    # ============================================================
    plt.figure()
    plt.plot(log["epoch"], log["train_acc"], label="Train")
    plt.plot(log["epoch"], log["test_acc"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(log["epoch"], log["loss_total"], label="Total")
    plt.plot(log["epoch"], log["loss_cls"], label="Cls")
    plt.plot(log["epoch"], log["loss_recon"], label="Recon")
    plt.plot(log["epoch"], log["loss_sparse"], label="Sparse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    print("Training Finished and logs saved.")
    
    return save_dir


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    args = parser.parse_args()
    CONFIG_PATH = args.config
    params = load_config(CONFIG_PATH)
    
    # デフォルト動作 (引数なし)
    train(device=DEVICE, params=params, save_model_name="model")
