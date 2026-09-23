import argparse
import json
import os

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from dataset_factory import load_multispectral_tensor
from models.model_factory import get_model


def infer(model_path, config_path, image_path, device="cuda", output_dir=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    class_names = config["class_names"]
    in_channels = int(config["in_channels"])
    model = get_model(config["architecture"], len(class_names), in_channels, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    image = load_multispectral_tensor(image_path, in_channels)
    image = TF.resize(image, config.get("input_size", [224, 224]),
                      interpolation=InterpolationMode.BILINEAR, antialias=True)
    normalization = config["normalization"]
    image = TF.normalize(image, normalization["mean"], normalization["std"]).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(image), dim=1)[0]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{name}.txt")
        sorted_values, sorted_indices = torch.sort(probability, descending=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("All Class Probabilities:\n")
            handle.write("=" * 50 + "\n\n")
            for value, index in zip(sorted_values, sorted_indices):
                score = value.item()
                handle.write(f"{class_names[index.item()]}: {score:.4f} ({score * 100:.2f}%)\n")
        print(f"Saved inference result to {output_path}")
    return probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    infer(args.model_path, args.config, args.image, args.device, args.output_dir)
