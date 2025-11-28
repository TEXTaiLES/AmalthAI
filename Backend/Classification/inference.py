import torch
from torchvision import transforms
from PIL import Image
import argparse
import json
from models.model_factory import get_model
import os

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def load_class_names(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def infer(model_path, model_name, class_names_path, image_path, device='cuda', output_dir=None):
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)

    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor = load_image(image_path).to(device)

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
            f.write("\n" + "="*50 + "\n")
            f.write("All Class Probabilities:\n")
            f.write("="*50 + "\n\n")
            
            # Sort probabilities in descending order
            probs_sorted = torch.sort(prob[0], descending=True)
            for i in range(num_classes):
                idx = probs_sorted.indices[i].item()
                prob_value = probs_sorted.values[i].item()
                f.write(f"{class_names[idx]}: {prob_value:.4f} ({prob_value*100:.2f}%)\n")
        
        print(f"Saved inference result to {output_path}")

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