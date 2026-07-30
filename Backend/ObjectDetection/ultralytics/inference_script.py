from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="frontend inputs")

parser.add_argument(
    "--checkpoint", 
    type=str, 
    default="yolo11n.pt",
    help="weights"
)
parser.add_argument(
    "--input", 
    type=str, 
    default="/path/to/image",
    help="Input image"
)

parser.add_argument(
    "--output", 
    type=str, 
    default="/path/to/result",
    help="Output image"
)
args = parser.parse_args()

model_selection = args.checkpoint
output = args.output
input = args.input

model = YOLO(model_selection)

model.predict(
    input,
    save=True,
    project= output
)

