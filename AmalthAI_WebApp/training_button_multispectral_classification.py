import argparse
import datetime
import os

from utils.comparensave import comparensave
from utils.experiment_organize import conduct_experiment_cls
from utils.load_config import load_config

config = load_config("config.yml")
BASE_HOST_PATH = config.get("paths").get("base_host_path")

parser = argparse.ArgumentParser(description="frontend inputs")
parser.add_argument("--model", default="ResNet18")
parser.add_argument("--user", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--in_channels", type=int, required=True)
parser.add_argument("--lr_left", type=float, default=0.001)
parser.add_argument("--lr_right", type=float, default=0.1)
parser.add_argument("--bs_left", type=int, default=16)
parser.add_argument("--bs_right", type=int, default=32)
parser.add_argument("--epoch_left", default="1")
parser.add_argument("--epoch_right", default="2")
for name in ("blur", "rotate", "flip", "scale", "dataset_already_split", "transfer_learning"):
    parser.add_argument(f"--{name}", default="false" if name != "transfer_learning" else "true")
args = parser.parse_args()

dataset = os.path.join("/multispectral_datasets", args.dataset)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
models = ("ResNet18", "ResNet50", "ConvNeXt", "EfficientNetB0", "MobileNetV2", "ShuffleNetV2")
selected = models if args.model == "allmodels" else (args.model,)

results = [conduct_experiment_cls(
    model, timestamp, dataset, args.lr_left, args.lr_right,
    args.bs_left, args.bs_right, args.epoch_left, args.epoch_right,
    args.blur, args.rotate, args.flip, args.scale,
    args.dataset_already_split, args.user, args.transfer_learning,
    multispectral=True, in_channels=args.in_channels,
) for model in selected]

if not all(result == "Succeeded" for result in results):
    raise SystemExit(1)

base_path = f"{BASE_HOST_PATH}/{args.user}/MultispectralClassification/runs/"
csv_path = os.path.join(base_path, "user_experiments.csv")
experiment_path = os.path.join(base_path, timestamp)
os.makedirs(experiment_path, exist_ok=True)
result = comparensave(experiment_path, timestamp, csv_path, dataset, "MsCls", maximize=True)
raise SystemExit(0 if result else 1)
