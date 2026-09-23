import argparse
import datetime
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset_factory import DatasetFactory
from models.model_factory import get_model


def train(config):
    factory = DatasetFactory(
        in_channels=config.in_channels,
        batch_size=config.batch_size,
        val_split=0.3,
        blur=config.blur,
        flip=config.flip,
        rotate=config.rotate,
        scale=config.scale,
        dataset_already_split=config.dataset_already_split,
    )
    train_loader, val_loader, num_classes, class_names, mean, std = factory.get_dataset(config.dataset)
    model = get_model(config.model, num_classes, config.in_channels, config.transfer_learning).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    metadata_path = os.path.join(config.dataset, "multispectral_metadata.json")
    metadata = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    save_dir = os.path.join(
        "/multispectral_classsave/runs", config.save_path, config.model,
        f"saved_{datetime.datetime.now().microsecond}",
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    model_config = {
        "schema_version": 1,
        "mode": "multispectral_classification",
        "architecture": config.model,
        "in_channels": config.in_channels,
        "band_names": metadata.get("band_names", [f"Band {i + 1}" for i in range(config.in_channels)]),
        "class_names": class_names,
        "input_size": [224, 224],
        "normalization": {"mean": mean, "std": std},
    }
    config_path = os.path.join(save_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)

    best_acc = -1.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                predicted = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"{best_acc:.2f}\n")
            print(f"accuracy={best_acc:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multispectral Classification Framework")
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default="checkpoints")
    for name in ("blur", "flip", "rotate", "scale", "dataset_already_split", "transfer_learning"):
        parser.add_argument(f"--{name}", default="false" if name != "transfer_learning" else "true",
                            choices=["true", "false"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name in ("blur", "flip", "rotate", "scale", "dataset_already_split", "transfer_learning"):
        setattr(args, name, getattr(args, name).lower() == "true")
    train(args)
