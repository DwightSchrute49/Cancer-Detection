import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LungCancerCNN


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    args = checkpoint.get("args", {})
    num_classes = len(class_to_idx)
    model = LungCancerCNN(
        num_classes=num_classes,
        backbone=args.get("backbone", "efficientnet_b0"),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_to_idx, args


def resolve_test_dir(data_dir: str) -> Path:
    root = Path(data_dir)
    if (root / "test").exists():
        return root / "test"
    return root


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained lung cancer CNN")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root or folder with a test subfolder")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("outputs", "best_model.pt"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--report-path", type=str, default=os.path.join("outputs", "evaluation_report.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_to_idx, checkpoint_args = load_checkpoint(args.checkpoint, device)

    image_size = int(checkpoint_args.get("image_size", args.image_size))
    test_dir = resolve_test_dir(args.data_dir)
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transform(image_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    idx_to_class = {index: label for label, index in class_to_idx.items()}
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true_names = [idx_to_class[index] for index in y_true]
    y_pred_names = [idx_to_class[index] for index in y_pred]

    accuracy = accuracy_score(y_true_names, y_pred_names)
    report = classification_report(y_true_names, y_pred_names, output_dict=True)
    matrix = confusion_matrix(y_true_names, y_pred_names, labels=list(class_to_idx.keys()))

    os.makedirs(Path(args.report_path).parent, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "classes": class_to_idx,
            },
            handle,
            indent=2,
        )

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_true_names, y_pred_names))
    print("Confusion matrix:\n", matrix)
    print(f"Saved report to {args.report_path}")


if __name__ == "__main__":
    main()
