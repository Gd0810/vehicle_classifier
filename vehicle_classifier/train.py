"""
Vehicle Type Classification - Model Training Script
====================================================
Usage:
    python train.py --images_dir ./images --csv _classes.csv --epochs 20

Place your images inside the 'images/' folder (same names as in the CSV).
The script will train a ResNet-50 model and save it to models/vehicle_classifier.pth
along with training plots in static/images/.
"""

import os
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 0           # set to 2+ on Linux/Mac for faster data loading
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Ambulance", "Box Truck", "Bus", "Bus- Small", "Concrete Mixer",
    "Construction Equipment", "Cyclist", "Fire Truck", "Garbage Truck",
    "Hatchback", "Motorbike", "Pickup", "Pickup- Utility", "SUV",
    "Sedan", "Tow Truck", "Tractor Trailer", "Trailer",
    "Truck- 2-Axle", "Truck- Multi-Axle", "Van"
]
NUM_CLASSES = len(CLASSES)


# ── Dataset ────────────────────────────────────────────────────────────────────
class VehicleDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = row["filename"].strip()
        img_path = os.path.join(self.images_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label_cols = [c for c in row.index if c != "filename"]
        label      = torch.tensor(row[label_cols].values.astype(np.float32))
        return image, label


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze layer4 + fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid(),
    )
    return model


# ── Helpers ────────────────────────────────────────────────────────────────────
def save_plot(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    axes[0].plot(epochs, history["train_loss"], color="#58a6ff", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   color="#f78166", linewidth=2, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#21262d", labelcolor="white")

    axes[1].plot(epochs, history["val_acc"], color="#3fb950", linewidth=2, label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(facecolor="#21262d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"[INFO] Saved training curves → {save_dir}/training_curves.png")


def save_class_dist(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    class_counts = df[CLASSES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(class_counts)))
    bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="#0d1117", linewidth=0.5)

    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(val)), ha="center", va="bottom", color="white", fontsize=8)

    ax.set_title("Dataset Class Distribution", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Vehicle Type", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.set_xticklabels(class_counts.index, rotation=45, ha="right", color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[INFO] Saved class distribution → {save_dir}/class_distribution.png")


def save_confusion_matrix(y_true, y_pred, class_names, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # Convert multi-hot to argmax for a compact CM
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 8})
    ax.set_title("Confusion Matrix", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=120,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[INFO] Saved confusion matrix → {save_dir}/confusion_matrix.png")


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="  Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).all(dim=1).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="  Val  "):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
            total   += images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    return total_loss / total, correct / total, all_labels, all_preds


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n{'='*60}")
    print(" Vehicle Type Classifier — Training")
    print(f"{'='*60}")
    print(f"  Device     : {DEVICE}")
    print(f"  CSV        : {args.csv}")
    print(f"  Images dir : {args.images_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"{'='*60}\n")

    # ── Load CSV ───────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    df["filename"] = df["filename"].str.strip()

    # Only keep rows where the image actually exists
    df["_exists"] = df["filename"].apply(
        lambda f: os.path.isfile(os.path.join(args.images_dir, f))
    )
    missing = (~df["_exists"]).sum()
    if missing:
        print(f"[WARN] {missing} images not found in '{args.images_dir}' — skipping them.")
    df = df[df["_exists"]].drop(columns="_exists").reset_index(drop=True)
    print(f"[INFO] Usable samples: {len(df)}")

    if len(df) == 0:
        print("[ERROR] No images found. Make sure your images are in the folder you specified.")
        sys.exit(1)

    # ── Class distribution plot ────────────────────────────────────────────────
    save_class_dist(df, args.plot_dir)

    # ── Train/Val split ────────────────────────────────────────────────────────
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[INFO] Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_tf, val_tf = get_transforms()
    train_ds = VehicleDataset(train_df, args.images_dir, train_tf)
    val_ds   = VehicleDataset(val_df,   args.images_dir, val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ── Training loop ──────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, y_true, y_pred = val_epoch(model, val_dl, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        flag = " ← best" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, "vehicle_classifier.pth"))

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  ({elapsed:.1f}s){flag}")

    # ── Post-training artifacts ────────────────────────────────────────────────
    save_plot(history, args.plot_dir)
    save_confusion_matrix(y_true, y_pred, CLASSES, args.plot_dir)

    # Save class names & training meta
    meta = {
        "classes"      : CLASSES,
        "num_classes"  : NUM_CLASSES,
        "img_size"     : IMG_SIZE,
        "best_val_acc" : round(best_val_acc * 100, 2),
        "epochs"       : args.epochs,
        "train_samples": len(train_df),
        "val_samples"  : len(val_df),
    }
    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  Training complete!  Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Model saved → {args.model_dir}/vehicle_classifier.pth")
    print(f"   Plots saved → {args.plot_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Type Classifier")
    parser.add_argument("--images_dir", default="./images",
                        help="Path to folder containing training images")
    parser.add_argument("--csv",        default="./_classes.csv",
                        help="Path to _classes.csv")
    parser.add_argument("--epochs",     type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--model_dir",  default="./models",
                        help="Directory to save the trained model")
    parser.add_argument("--plot_dir",   default="./static/images",
                        help="Directory to save training plots")
    args = parser.parse_args()
    main(args)
