import argparse
import datetime
import json
import os

from utils.comparensave import comparensave
from utils.experiment_organize import conduct_experiment_cls
from utils.load_config import load_config

parser = argparse.ArgumentParser(description="Multispectral classification frontend inputs")
parser.add_argument("--model", default="ResNet18")
parser.add_argument("--user", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--lr_left", type=float, default=0.001)
parser.add_argument("--lr_right", type=float, default=0.1)
parser.add_argument("--bs_left", type=int, default=16)
parser.add_argument("--bs_right", type=int, default=32)
parser.add_argument("--epoch_left", default="1")
parser.add_argument("--epoch_right", default="2")
parser.add_argument("--blur", default="false")
parser.add_argument("--rotate", default="false")
parser.add_argument("--flip", default="false")
parser.add_argument("--scale", default="false")
parser.add_argument("--dataset_already_split", default="false")
parser.add_argument("--transfer_learning", default="true")
args = parser.parse_args()

config = load_config("config.yml")
base_host_path = config.get("paths").get("base_host_path")
dataset = os.path.join("/multispectral_datasets", args.dataset)
metadata_path = os.path.join(base_host_path, args.user, "Datasets",
                             "Multispectral-Classification", args.dataset,
                             "multispectral_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as handle:
    in_channels = int(json.load(handle)["num_channels"])
local_dataset_path = os.path.dirname(metadata_path)
dataset_already_split = (os.path.isdir(os.path.join(local_dataset_path, "train")) and
                         os.path.isdir(os.path.join(local_dataset_path, "val")))

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
models = ("ResNet18", "ResNet50", "ConvNeXt", "EfficientNetB0", "MobileNetV2", "ShuffleNetV2")


def run(model):
    return conduct_experiment_cls(
        model, timestamp, dataset, args.lr_left, args.lr_right,
        args.bs_left, args.bs_right, args.epoch_left, args.epoch_right,
        args.blur, args.rotate, args.flip, args.scale,
        str(dataset_already_split).lower(), args.user, args.transfer_learning,
        in_channels=in_channels,
    )


if args.model == "allmodels":
    results = [run(model) for model in models]
    final_result = "Succeeded" if all(result == "Succeeded" for result in results) else "Failed"
else:
    final_result = run(args.model)

if final_result != "Succeeded":
    raise SystemExit(1)

base_path = os.path.join(base_host_path, args.user, "MultispectralClassification", "runs")
experiment_path = os.path.join(base_path, timestamp)
os.makedirs(experiment_path, exist_ok=True)
result = comparensave(
    experiment_path, timestamp, os.path.join(base_path, "user_experiments.csv"),
    dataset, "MsCls", maximize=True,
)
raise SystemExit(0 if result else 1)
