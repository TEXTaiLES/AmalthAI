import torch
from torchvision import transforms
from PIL import Image
import argparse
import json
from models.model_factory import get_model
import os
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    rgb_image = image.resize((224, 224))
    rgb_image = np.asarray(rgb_image).astype(np.float32) / 255.0

    return image_tensor, rgb_image


def load_class_names(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_target_layer(model, model_name):
    model_name = model_name.lower()

    if "resnet18" in model_name:
        return model.layer4[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    elif "mobilenetv2" in model_name:
        return model.features[-1]
    elif "shufflenet" in model_name:
        return model.conv5
    elif "resnet50" in model_name:
        return model.layer4[-1]
    elif "convnext" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def infer(model_path, model_name, class_names_path, image_path, device='cuda', output_dir=None):
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)

    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor, rgb_image = load_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        pred_idx = prob.argmax(dim=1).item()
        confidence = prob.max().item()
        pred_class_name = class_names[pred_idx]

    result_text = f"Predicted Class: {pred_class_name} | Confidence: {confidence:.2f}"
    print(result_text)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}.txt")

        with open(output_path, "w") as f:
            f.write(f"Predicted Class: {pred_class_name} | Confidence: {confidence:.4f}\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("All Class Probabilities:\n")
            f.write("=" * 50 + "\n\n")

            probs_sorted = torch.sort(prob[0], descending=True)
            for i in range(num_classes):
                idx = probs_sorted.indices[i].item()
                prob_value = probs_sorted.values[i].item()
                f.write(f"{class_names[idx]}: {prob_value:.4f} ({prob_value * 100:.2f}%)\n")
        print(f"Saved inference result to {output_path}")

        # ---------------- Grad-CAM ----------------

        target_layer = get_target_layer(model, model_name)

        cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )

        grayscale_cam = cam(input_tensor=image_tensor)[0]

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        gradcam_path = os.path.join(output_dir, f"{image_name}_gradcam.jpg")
        Image.fromarray(visualization).save(gradcam_path)

        print(f"Saved Grad-CAM to {gradcam_path}")

    return pred_class_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained .pth file')
    parser.add_argument('--model', type=str, required=True, help='Model architecture used')
    parser.add_argument('--class_names', type=str, required=True, help='Path to class_names.json')
    parser.add_argument('--image', type=str, required=True, help='Image path to infer')
    parser.add_argument('--output_dir', type=str, help='Directory to save inference result')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cpu or cuda')


    args = parser.parse_args()
    infer(args.model_path, args.model, args.class_names, args.image, args.device, args.output_dir)