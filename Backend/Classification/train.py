import torch
import torch.nn as nn
import torch.optim as optim
from models.model_factory import get_model
from tqdm import tqdm
import argparse
from config import get_config
import os
import datetime
from dataset_factory import DatasetFactory
import json


def train(config):
    factory = DatasetFactory(
        batch_size=config.batch_size, 
        val_split=0.3,
        blur=config.blur,
        flip=config.flip,
        rotate=config.rotate
    )
    train_loader, val_loader, num_classes, class_names = factory.get_dataset(config.dataset)
    model = get_model(config.model, num_classes).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_acc = 0.0

    current_time_micro = datetime.datetime.now().microsecond
    save_dir = os.path.join("/classification/runs", config.save_path, config.model, f"saved_{current_time_micro}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"accuracy={best_acc:.2f}")
            print(f"Saved Best Model with Accuracy: {best_acc:.2f}% at {save_path}")

            # Save the score to result.txt
            result_path = os.path.join(save_dir, "result.txt")
            with open(result_path, "w") as f:
                f.write(f"{best_acc:.2f}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification Framework")

    parser.add_argument('--model', type=str, default='ResNet18', help='Model backbone')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Base path to save model weights')
    
    # Augmentation arguments
    parser.add_argument('--blur', type=str, default='false', choices=['true', 'false'], help='Enable Gaussian blur augmentation')
    parser.add_argument('--flip', type=str, default='false', choices=['true', 'false'], help='Enable horizontal flip augmentation')
    parser.add_argument('--rotate', type=str, default='false', choices=['true', 'false'], help='Enable rotation augmentation')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Convert string boolean values to actual booleans
    args.blur = args.blur.lower() == 'true'
    args.flip = args.flip.lower() == 'true'
    args.rotate = args.rotate.lower() == 'true'

    config = get_config(args)
    train(config)