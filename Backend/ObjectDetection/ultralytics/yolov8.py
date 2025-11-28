from ultralytics import YOLO
import argparse
import datetime
import pandas as pd

# Arguments for the parser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--bs', type=int, required=True, help='Batch Size')
parser.add_argument('--lr', type=float, required=True, help='Learning Rate')
parser.add_argument('--dataset',type = str, required=True, help = "Dataset Config File")
parser.add_argument('--timestamp',type = str, required=True, help = "Timestamp")

parser.add_argument(
    "--hue", 
    type=float,
    default=0.5,
    help="Hue augmentation factor"
)
parser.add_argument(
    "--saturation",
    type=float,
    default=0.5,
    help="Saturation augmentation factor"
)
parser.add_argument(
    "--value",          
    type=float,
    default=0.5,
    help="Value augmentation factor"
)
parser.add_argument(
    "--rotate", 
    type=int,
    default=15,
    help="Rotate augmentation factor"
)
parser.add_argument(
    "--flip", 
    type=float,
    default=0.5,
    help="Flip augmentation factor"
)   

args = parser.parse_args()

# Load a model
model = YOLO("/yolo/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

current_time_micro = datetime.datetime.now().microsecond
# Start training
results = model.train(data= args.dataset, epochs=args.epochs, batch = args.bs,lr0 = args.lr, imgsz=460, device = [0],project=f"/yolo/runs/{args.timestamp}/YOLOv8", name=f"run_{current_time_micro}",hsv_h = args.hue, hsv_s = args.saturation, hsv_v = args.value, degrees = args.rotate, fliplr = args.flip)

df = pd.read_csv(f"/yolo/runs/{args.timestamp}/YOLOv8/run_{current_time_micro}/results.csv")

# Locate and print the required metric to help katib
map5095 = df["metrics/mAP50-95(B)"].iloc[-1]
print(f"map5095={map5095:.4f}")

result_path = f"/yolo/runs/{args.timestamp}/YOLOv8/run_{current_time_micro}/result.txt"
with open(result_path, "w") as f:
    f.write(f"{map5095:.4f}\n")
