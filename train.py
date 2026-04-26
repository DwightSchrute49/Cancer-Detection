import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import LungCancerCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def extract_targets(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return [dataset.dataset.targets[index] for index in dataset.indices]
    return list(dataset.targets)


def compute_class_weights(dataset, class_to_idx: dict) -> torch.Tensor:
    targets = extract_targets(dataset)
    class_counts = torch.bincount(torch.tensor(targets), minlength=len(class_to_idx)).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return weights


def validate_imagefolder_layout(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Create it and add class folders."
        )

    class_dirs = [item for item in dataset_dir.iterdir() if item.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            "No class folders were found inside the dataset directory. "
            "ImageFolder expects a structure like:\n"
            "data/\n"
            "  class_a/\n"
            "  class_b/\n\n"
            "Or, for a split dataset:\n"
            "data/\n"
            "  train/\n"
            "    class_a/\n"
            "    class_b/\n"
            "  test/\n"
            "    class_a/\n"
            "    class_b/"
        )


def make_datasets(data_dir: str, image_size: int, val_split: float, seed: int):
    train_transform, eval_transform = build_transforms(image_size)
    root = Path(data_dir)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if train_dir.exists() and test_dir.exists():
        validate_imagefolder_layout(train_dir)
        validate_imagefolder_layout(test_dir)
        full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        if val_dir.exists():
            validate_imagefolder_layout(val_dir)
            train_dataset = full_train_dataset
            val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        else:
            val_size = max(1, int(val_split * len(full_train_dataset)))
            train_size = len(full_train_dataset) - val_size
            if train_size <= 0:
                raise ValueError("Training set is too small for the requested validation split.")
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
            val_dataset.dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        return train_dataset, val_dataset, test_dataset

    validate_imagefolder_layout(root)
    full_dataset = datasets.ImageFolder(root, transform=train_transform)
    test_size = max(1, int(0.15 * len(full_dataset)))
    val_size = max(1, int(val_split * len(full_dataset)))
    train_size = len(full_dataset) - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested val/test split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    val_dataset.dataset = datasets.ImageFolder(root, transform=eval_transform)
    test_dataset.dataset = datasets.ImageFolder(root, transform=eval_transform)
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(outputs, labels)
        total_batches += 1

    return total_loss / max(1, total_batches), total_acc / max(1, total_batches)


def save_checkpoint(output_dir: str, model, class_to_idx, args, epoch: int, val_acc: float):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "val_acc": val_acc,
    }
    torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for lung cancer image classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root or folder with train/test subfolders")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
        help="Pretrained backbone to use",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet18 weights")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = make_datasets(args.data_dir, args.image_size, args.val_split, args.seed)

    if isinstance(train_dataset, torch.utils.data.Subset):
        class_to_idx = train_dataset.dataset.class_to_idx
    else:
        class_to_idx = train_dataset.class_to_idx

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    num_classes = len(class_to_idx)
    model = LungCancerCNN(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    class_weights = compute_class_weights(train_dataset, class_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(args.output_dir, model, class_to_idx, args, epoch, val_acc)

        scheduler.step(val_acc)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, scaler=None, train=False)
    print(f"Best epoch: {best_epoch} | Best val_acc: {best_val_acc:.4f}")
    print(f"Test  loss: {test_loss:.4f} | Test  acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
