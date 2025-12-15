import os
import csv
from datetime import datetime
from PIL import Image


# function to find the registered datasets
def load_datasets(filepath, mode):
    datasets_path = filepath
    collections = []

    for name in os.listdir(datasets_path):
        dataset_folder = os.path.join(datasets_path, name)

        if mode == "Seg":
            images_folder = os.path.join(dataset_folder, "images", "train")
            if os.path.isdir(images_folder) and not name.startswith('.'):
                num_items = len(os.listdir(images_folder))
                collections.append({
                    "id"         : name,
                    "name"       : name,
                    "num_samples": num_items
                })

        elif mode == "OD":
            images_folder = os.path.join(dataset_folder, "train", "images")
            if os.path.isdir(images_folder) and not name.startswith('.'):
                num_items = len(os.listdir(images_folder))
                collections.append({
                    "id"         : name,
                    "name"       : name,
                    "num_samples": num_items
                })

        elif mode == "Cls":
            if os.path.isdir(dataset_folder) and not name.startswith('.'):
                num_items = 0
                for cls in os.listdir(dataset_folder):
                    cls_path = os.path.join(dataset_folder, cls)
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
        image_files = [
            f for f in os.listdir(images_folder)
            if os.path.isfile(os.path.join(images_folder, f))
        ]

        num_items = len(image_files)
        creation_timestamp = os.path.getctime(images_folder)
        date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
        
        dataset_info = {
            "name"     : name,
            "num_items": num_items,
            "date"     : date
        }

        dataset_items = [
            {
                "name" : f,
                "image": os.path.join(images_folder, f)
            }
            for f in image_files
        ]

    elif mode == "detection":
        images_folder = os.path.join(dataset_folder, "train", "images")
        image_files = [
            f for f in os.listdir(images_folder)
            if os.path.isfile(os.path.join(images_folder, f))
        ]

        num_items = len(image_files)
        creation_timestamp = os.path.getctime(images_folder)
        date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
        
        dataset_info = {
            "name"     : name,
            "num_items": num_items,
            "date"     : date
        }

        dataset_items = [
            {
                "name" : f,
                "image": os.path.join(images_folder, f)
            }
            for f in image_files
        ]
    
    elif mode == "classification":
        dataset_items = []
        if os.path.isdir(dataset_folder):
            for cls in os.listdir(dataset_folder):
                cls_path = os.path.join(dataset_folder, cls)
                if os.path.isdir(cls_path):
                    image_files = [
                        f for f in os.listdir(cls_path)
                        if os.path.isfile(os.path.join(cls_path, f))
                    ]
                    for f in image_files:
                        dataset_items.append({
                            "name" : f,
                            "image": os.path.join(cls_path, f),
                            "class": cls
                        })

        num_items = len(dataset_items)
        creation_timestamp = os.path.getctime(dataset_folder)
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
