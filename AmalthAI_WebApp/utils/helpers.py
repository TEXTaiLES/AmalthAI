import os
import csv
import subprocess
from datetime import datetime
from PIL import Image

# prefer filesystem birth time when available else fall back to mtime.
def get_best_timestamp(path):
    try:
        stat_result = os.stat(path)
        birth = getattr(stat_result, "st_birthtime", None)
        if birth:
            return birth
    except OSError:
        return os.path.getmtime(path)

    try:
        proc = subprocess.run(
            ["stat", "-c", "%W", path],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            value = int(proc.stdout.strip())
            if value > 0:
                return value
    except (ValueError, OSError):
        pass

    return os.path.getmtime(path)


# function to find the registered datasets
def load_datasets(filepath, mode):
    datasets_path = filepath
    collections = []

    for name in os.listdir(datasets_path):
        dataset_folder = os.path.join(datasets_path, name)

        if mode == "Seg":
            images_folder = os.path.join(dataset_folder, "images", "train")
            val_folder = os.path.join(dataset_folder, "images", "val")
            if os.path.isdir(images_folder) and not name.startswith('.'):
                num_items = len(os.listdir(images_folder))
                if os.path.isdir(val_folder):
                    num_items += len(os.listdir(val_folder))
                collections.append({
                    "id"         : name,
                    "name"       : name,
                    "num_samples": num_items
                })

        elif mode == "OD":
            images_folder = os.path.join(dataset_folder, "train", "images")
            val_folder = os.path.join(dataset_folder, "valid", "images")
            if os.path.isdir(images_folder) and not name.startswith('.'):
                num_items = len(os.listdir(images_folder))
                if os.path.isdir(val_folder):
                    num_items += len(os.listdir(val_folder))
                collections.append({
                    "id"         : name,
                    "name"       : name,
                    "num_samples": num_items
                })

        elif mode == "Cls":
            if os.path.isdir(dataset_folder) and not name.startswith('.'):
                train_dir = os.path.join(dataset_folder, "train")
                val_dir = os.path.join(dataset_folder, "val")
                use_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)

                num_items = 0
                base_dirs = [train_dir, val_dir] if use_split else [dataset_folder]
                for base_dir in base_dirs:
                    for cls in os.listdir(base_dir):
                        cls_path = os.path.join(base_dir, cls)
                        if os.path.isdir(cls_path):
                            num_items += len(os.listdir(cls_path))
                collections.append({
                    "id"         : name,
                    "name"       : name,
                    "num_samples": num_items
                })

    return collections



def load_dataset_info(filepath, name, mode):
    dataset_folder = os.path.join(filepath, name)

    dataset_info  = {}
    dataset_items = []

    if mode == "segmentation":
        images_folder = os.path.join(dataset_folder, "images", "train")
        val_folder = os.path.join(dataset_folder, "images", "val")
        train_files = [
            f for f in os.listdir(images_folder)
            if os.path.isfile(os.path.join(images_folder, f))
        ]
        val_files = []
        if os.path.isdir(val_folder):
            val_files = [
                f for f in os.listdir(val_folder)
                if os.path.isfile(os.path.join(val_folder, f))
            ]

        num_items = len(train_files) + len(val_files)
        creation_timestamp = get_best_timestamp(images_folder)
        date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
        
        dataset_info = {
            "name"     : name,
            "num_items": num_items,
            "date"     : date
        }

        dataset_items = [
            {
                "name" : f,
                "image": os.path.join(images_folder, f),
                "split": "train"
            }
            for f in train_files
        ]
        dataset_items += [
            {
                "name" : f,
                "image": os.path.join(val_folder, f),
                "split": "val"
            }
            for f in val_files
        ]

    elif mode == "detection":
        images_folder = os.path.join(dataset_folder, "train", "images")
        val_folder = os.path.join(dataset_folder, "valid", "images")
        train_files = [
            f for f in os.listdir(images_folder)
            if os.path.isfile(os.path.join(images_folder, f))
        ]
        val_files = []
        if os.path.isdir(val_folder):
            val_files = [
                f for f in os.listdir(val_folder)
                if os.path.isfile(os.path.join(val_folder, f))
            ]

        num_items = len(train_files) + len(val_files)
        creation_timestamp = get_best_timestamp(images_folder)
        date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
        
        dataset_info = {
            "name"     : name,
            "num_items": num_items,
            "date"     : date
        }

        dataset_items = [
            {
                "name" : f,
                "image": os.path.join(images_folder, f),
                "split": "train"
            }
            for f in train_files
        ]
        dataset_items += [
            {
                "name" : f,
                "image": os.path.join(val_folder, f),
                "split": "val"
            }
            for f in val_files
        ]
    
    elif mode == "classification":
        dataset_items = []
        if os.path.isdir(dataset_folder):
            train_dir = os.path.join(dataset_folder, "train")
            val_dir = os.path.join(dataset_folder, "val")
            use_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)
            base_dirs = [(train_dir, "train"), (val_dir, "val")] if use_split else [(dataset_folder, None)]

            for base_dir, split_label in base_dirs:
                for cls in os.listdir(base_dir):
                    cls_path = os.path.join(base_dir, cls)
                    if os.path.isdir(cls_path):
                        image_files = [
                            f for f in os.listdir(cls_path)
                            if os.path.isfile(os.path.join(cls_path, f))
                        ]
                        for f in image_files:
                            item = {
                                "name" : f,
                                "image": os.path.join(cls_path, f),
                                "class": cls
                            }
                            if split_label:
                                item["split"] = split_label
                            dataset_items.append(item)

        num_items = len(dataset_items)
        creation_timestamp = get_best_timestamp(dataset_folder)
        date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
        
        dataset_info = {
            "name"     : name,
            "num_items": num_items,
            "date"     : date
        }

    return dataset_info, dataset_items



# Shows the available models for each task which are registered in the CSV files
def load_models_available(filepath):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]  



# Returns the trained models for each task from the CSV files
def load_models(csv_file):
    models = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            row['id'] = i + 1
            try:
                row['score'] = float(row['score'])
            except ValueError:
                row['score'] = None
            models.append(row)
    return models


def get_max_image_size(items):
    max_w, max_h = 0, 0
    for it in items:
        try:
            img = Image.open(it["image"])
            w, h = img.size
            if w > max_w: max_w = w
            if h > max_h: max_h = h
        except:
            pass
    return max_w, max_h
