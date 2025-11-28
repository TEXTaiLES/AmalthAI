import yaml
import os
import subprocess
import datetime
import copy
import shutil
import kubeflow.katib as katib
import random
from utils.comparensave import comparensave
from utils.experiment_organize import conduct_experiment_od
import argparse
from utils.load_config import load_config

# Yaml config
config = load_config("config.yml")
BASE_HOST_PATH = config.get("paths").get("base_host_path")

# Argument parser for frontend inputs
parser = argparse.ArgumentParser(description="frontend inputs")

parser.add_argument(
    "--model", 
    type=str, 
    default="YOLOv11",
    help="Prefered model"
)
parser.add_argument(
    "--dataset", 
    type=str, 
    default="/yolo/........",
    help="Dataset selection"
)

parser.add_argument(
    "--lr_left", 
    type=float,
    default=0.001,
    help="Learning rate left range"
)

parser.add_argument(
    "--lr_right", 
    type=float,
    default=0.1,
    help="Learning rate right range"
)

parser.add_argument(
    "--bs_left", 
    type=int,
    default=2,
    help="Batch Size left range"
)

parser.add_argument(
    "--bs_right", 
    type=int,
    default=3,
    help="Batch Size right range"
)

parser.add_argument(
    "--epoch_left", 
    type=str,
    default="1",
    help="Epochs left range"
)

parser.add_argument(
    "--epoch_right", 
    type=str,
    default="2",
    help="Epochs right range"
)

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
model_selection = args.model
dataset = os.path.join("/objdet_datasets", args.dataset)
dataset = os.path.join(dataset, "data.yaml")
print(dataset)
print(model_selection)
lr_right = args.lr_right
lr_left = args.lr_left
bs_right = args.bs_right
bs_left = args.bs_left
epoch_right = args.epoch_right
epoch_left = args.epoch_left

hue = args.hue
saturation = args.saturation
value = args.value
rotate = args.rotate
flip = args.flip

def get_timestamp_path():
    now = datetime.datetime.now()
    timestamp_path = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_path

# define timestamp for the current run
timestamp_path = get_timestamp_path()

# Conduct experiments based on model selection
models_list = ("yolov8", "yolov11")

if model_selection == "allmodels":
    all_success = True  # flag

    for model in models_list:
        res = conduct_experiment_od(model, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epoch_left, epoch_right, hue, saturation, value, rotate, flip)
        print(f"Last condition for {model}: {res}")

        if res != "Succeeded":
            all_success = False

    final_res = "Succeeded" if all_success else "Failed"

else:
    final_res = conduct_experiment_od(model_selection, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epoch_left, epoch_right, hue, saturation, value, rotate, flip)
    print(f"Last condition for {model_selection}: {final_res}")


if final_res == "Succeeded":
    print("Experiment completed successfully.")
    base_path = f"{BASE_HOST_PATH}/ObjectDetection/runs/"
    csv_path = f"{BASE_HOST_PATH}/ObjectDetection/runs/user_experiments.csv"
    experiment_path = os.path.join(base_path, timestamp_path)
    os.makedirs(experiment_path, exist_ok=True)
    
    # Post processing to find the best trial
    result = comparensave(experiment_path, timestamp_path, csv_path, dataset, "OD", maximize=True)

    if result:
        print("The best trial has the following attributes:")
        print("Model:", result['model'])
        print("Run:", result['run'])
        print("Score:", result['score'])
        print("Weights path:", result['weights'])
        print("Config path:", result['config'])
    else:
        print("No valid trial found.")

    exit(0)
else:
    print("Experiment did not complete successfully.")
    exit(1)