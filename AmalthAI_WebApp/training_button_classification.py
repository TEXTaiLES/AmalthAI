import yaml
import os
import subprocess
import datetime
import copy
import shutil
import kubeflow.katib as katib
import random
from utils.comparensave import comparensave
import argparse
from utils.experiment_organize import conduct_experiment_cls
from utils.load_config import load_config

# Yaml config
config = load_config("config.yml")
BASE_HOST_PATH = config.get("paths").get("base_host_path")

# Argument parser for frontend inputs
parser = argparse.ArgumentParser(description="frontend inputs")

parser.add_argument(
    "--model", 
    type=str, 
    default="ResNet18",
    help="Prefered model"
)
parser.add_argument(
    "--user",
    type=str,
    required=True,
    help="User slug for per-user paths"
)
parser.add_argument(
    "--dataset", 
    type=str, 
    default="/classification/FIDdataset",
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
    default=16,
    help="Batch Size left range"
)

parser.add_argument(
    "--bs_right", 
    type=int,
    default=32,
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
    "--blur", 
    type=str,
    default="false",
    help="Apply blur augmentation"
)

parser.add_argument(
    "--rotate", 
    type=str,
    default="false",
    help="Apply rotate augmentation"
)

parser.add_argument(
    "--flip", 
    type=str,
    default="false",
    help="Apply flip augmentation"
)

parser.add_argument(
    "--scale", 
    type=str,
    default="false",
    help="Apply scale augmentation"
)

parser.add_argument(
    "--dataset_already_split",
    type=str,
    default="false",
    help="Dataset already split into train/val"
)

# Sanity checks and parsing
args = parser.parse_args()
model_selection = args.model
user_slug = args.user
dataset = os.path.join("/class_datasets", args.dataset)
print(dataset)
print(model_selection)
lr_right = args.lr_right
lr_left = args.lr_left
bs_right = args.bs_right
bs_left = args.bs_left
epoch_right = args.epoch_right
epoch_left = args.epoch_left

blur = args.blur
rotate = args.rotate
flip = args.flip
scale = args.scale
dataset_already_split = args.dataset_already_split

def get_timestamp_path():
    now = datetime.datetime.now()
    timestamp_path = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_path

# define timestamp for the current run
timestamp_path = get_timestamp_path()

# Conduct experiments based on model selection
models_list = ("ResNet18", "EfficientNetB0","MobileNetV2","ShuffleNetV2")

if model_selection == "allmodels":
    all_success = True  # flag

    for model in models_list:
        res = conduct_experiment_cls(model, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epoch_left, epoch_right, blur, rotate, flip, scale, dataset_already_split, user_slug)
        print(f"Last condition for {model}: {res}")

        if res != "Succeeded":
            all_success = False

    final_res = "Succeeded" if all_success else "Failed"

else:
    final_res = conduct_experiment_cls(model_selection, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epoch_left, epoch_right, blur, rotate, flip, scale, dataset_already_split, user_slug)
    print(f"Last condition for {model_selection}: {final_res}")


if final_res == "Succeeded":
    print("Experiment completed successfully.")
    base_path = f"{BASE_HOST_PATH}/{user_slug}/Classification/runs/"
    csv_path = f"{BASE_HOST_PATH}/{user_slug}/Classification/runs/user_experiments.csv"
    experiment_path = os.path.join(base_path, timestamp_path)
    os.makedirs(experiment_path, exist_ok=True)

    # Post processing to find the best trial
    result = comparensave(experiment_path, timestamp_path, csv_path, dataset,"Cls",maximize=True)
    
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