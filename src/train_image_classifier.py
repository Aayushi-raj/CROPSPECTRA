"""
CROPSPECTRA — Autonomous Crop Inspection Rover
Phase: Image Classification Model Training
Model: MobileNetV3 (Transfer Learning)
Dataset: Rice Leaf Diseases (archive1)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def get_data_transforms(img_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_transform


def build_model(arch="mobilenet_v3_small", num_classes=6, pretrained=True):
    print(f"[*] Initializing {arch} (pretrained={pretrained}) for {num_classes} classes...")
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total
    val_acc = np.mean(np.array(all_preds) == np.array(all_targets))

    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)


def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'g-^', label='Validation Acc', linewidth=2)
    ax2.set_title('Training and Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Metrics plot saved to: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c.replace('_', ' ') for c in class_names])
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=30, values_format='d')
    ax.set_title('Validation Confusion Matrix - CROPSPECTRA', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CROPSPECTRA MobileNetV3 Leaf Disease Classifier")
    parser.add_argument("--train-dir", type=str, default="DATASET/archive1/RiceLeafsDisease/train")
    parser.add_argument("--val-dir", type=str, default="DATASET/archive1/RiceLeafsDisease/validation")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save plots and evaluation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    # Prepare datasets
    train_tf, val_tf = get_data_transforms(img_size=args.img_size)

    if not os.path.exists(args.train_dir) or not os.path.exists(args.val_dir):
        print(f"Error: Train directory '{args.train_dir}' or Val directory '{args.val_dir}' not found.")
        sys.exit(1)

    train_dataset = ImageFolder(args.train_dir, transform=train_tf)
    val_dataset = ImageFolder(args.val_dir, transform=val_tf)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"[*] Found {len(train_dataset)} training samples across {num_classes} classes.")
    print(f"[*] Found {len(val_dataset)} validation samples across {num_classes} classes.")
    print(f"[*] Classes: {class_names}")

    # Save class mappings
    class_mapping = {
        "classes": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()}
    }
    mapping_path = os.path.join(args.output_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=4)
    print(f"[+] Class mapping saved to {mapping_path}")

    # DataLoaders
    # On Windows, num_workers=0 avoids multiprocessing spawn issues
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer, Scheduler
    model = build_model(arch=args.arch, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_cropspectra_mobilenetv3.pth")

    print("\n" + "=" * 65)
    print("STARTING TRAINING PIPELINE")
    print("=" * 65)

    start_time = time.time()
    best_preds = None
    best_targets = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, targets = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_preds = preds
            best_targets = targets
            torch.save({
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "class_names": class_names,
                "arch": args.arch
            }, best_model_path)
            best_mark = " [NEW BEST]"
        else:
            best_mark = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%{best_mark}"
        )

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"[OK] Training Completed in {total_time/60:.2f} minutes.")
    print(f"[OK] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[OK] Saved best model to: {best_model_path}")
    print("=" * 65)

    # Plot metrics
    metrics_plot_path = os.path.join(args.reports_dir, "training_metrics.png")
    plot_metrics(history, metrics_plot_path)

    # Confusion matrix and classification report
    if best_preds is not None and best_targets is not None:
        cm = confusion_matrix(best_targets, best_preds)
        cm_plot_path = os.path.join(args.reports_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, cm_plot_path)

        report = classification_report(best_targets, best_preds, target_names=class_names, digits=4)
        print("\n=== FINAL VALIDATION CLASSIFICATION REPORT ===")
        print(report)

        report_txt_path = os.path.join(args.reports_dir, "classification_report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(f"CROPSPECTRA MobileNetV3 Training Summary\n")
            f.write(f"Total Epochs: {args.epochs}\n")
            f.write(f"Best Validation Accuracy: {best_val_acc*100:.2f}%\n\n")
            f.write(report)
        print(f"[+] Classification report saved to: {report_txt_path}")


if __name__ == "__main__":
    main()
