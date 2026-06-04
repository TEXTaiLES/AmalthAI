from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import tomllib
import math
import os
from werkzeug.utils import secure_filename
import subprocess
import glob
import csv
from datetime import datetime
import shutil
import threading
import json
import uuid
from utils.models_page import write_results
from utils.helpers import load_datasets, load_models_available, load_dataset_info, get_max_image_size, load_models, get_best_timestamp
from utils.load_config import load_config
import docker
import zipfile
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import requests


client = docker.from_env()

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config.from_file('config.toml', load=tomllib.load, text=False)

# Login requirements
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Lightweight in-memory user registry (per process)
_user_registry = {}


class User(UserMixin):
    def __init__(self, user_id, email=None, slug=None, full_name=None):
        self.id = user_id
        self.email = email
        self.slug = slug
        self.full_name = full_name

@login_manager.user_loader
def load_user(user_id):
    return _user_registry.get(str(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        DIRECTUS_URL = "http://textailes.athenarc.gr"
        resp = requests.post(
            f"{DIRECTUS_URL}/auth/login",
            json={
                "email": username,
                "password": password
            },
            headers={"Content-Type": "application/json"}
        )
        app.logger.info(f"Login response status: {resp.status_code}, body: {resp.text}")
        if resp.ok:
            slug = safe_user_slug(username)
            user = User(
                user_id=slug,
                email=username,
                slug=slug,
                full_name=f"@{slug}",
            )
            _user_registry[str(user.id)] = user
            ensure_user_folders(slug)
            login_user(user)
            return redirect(url_for("index"))

        flash("Wrong Credentials.")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# App configurations

# Yaml config
config = load_config("config.yml")

BASE_HOST_PATH      = config.get("paths").get("base_host_path")
BASE_HOST_PATH_OUT      = config.get("paths").get("base_host_path_out")
IMAGE_SEGM_CLS  = config.get("images").get("classification")
IMAGE_OD        = config.get("images").get("detection")


def safe_user_slug(email):
    base = email.split("@", 1)[0].strip().lower()
    return "".join(c for c in base if c.isalnum() or c in ("-", "_"))


def user_root(slug):
    return os.path.join(BASE_HOST_PATH, slug)


def get_current_user_slug():
    slug = getattr(current_user, "slug", None)
    return slug or "guest"


def ensure_model_db_file(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("name,trained_on,score,date,checkpoint_path,config_path\n")


def ensure_user_folders(slug):
    root = user_root(slug)
    paths = [
        os.path.join(root, "Datasets", "Segmentation"),
        os.path.join(root, "Datasets", "Object-Detection"),
        os.path.join(root, "Datasets", "Classification"),
        os.path.join(root, "Segmentation", "runs"),
        os.path.join(root, "ObjectDetection", "runs"),
        os.path.join(root, "Classification", "runs"),
        os.path.join(root, "models_db"),
        os.path.join(root, "exps"),
        os.path.join(root, "inference", "segmentation"),
        os.path.join(root, "inference", "detection"),
        os.path.join(root, "inference", "classification"),
        os.path.join(root, "tmp_datasets_zips"),
        os.path.join(root, "tmp_datasets"),
        os.path.join(root, "train_jobs"),
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)

    ensure_model_db_file(os.path.join(root, "models_db", "trained_models_db_segm.csv"))
    ensure_model_db_file(os.path.join(root, "models_db", "trained_models_db_od.csv"))
    ensure_model_db_file(os.path.join(root, "models_db", "trained_models_db_cls.csv"))


def _job_status_path(user_slug, job_id):
    return os.path.join(user_root(user_slug), "train_jobs", f"{job_id}.json")


def _write_job_status(user_slug, job_id, data):
    os.makedirs(os.path.dirname(_job_status_path(user_slug, job_id)), exist_ok=True)
    with open(_job_status_path(user_slug, job_id), "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _read_job_status(user_slug, job_id):
    path = _job_status_path(user_slug, job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_training_job(user_slug, job_id, cmd, mode, paths):
    try:
        process = subprocess.run(cmd)
        if process.returncode == 0:
            write_results(*paths[mode])
            status = {"status": "succeeded", "return_code": process.returncode}
        else:
            status = {"status": "failed", "return_code": process.returncode}
    except Exception as exc:
        status = {"status": "failed", "return_code": None, "error": str(exc)}

    _write_job_status(user_slug, job_id, status)


@app.route("/user-datasets/<path:filename>")
@login_required
def user_dataset_files(filename):
    user_slug = get_current_user_slug()
    data_dir = os.path.abspath(os.path.join(user_root(user_slug), "Datasets"))
    requested = os.path.abspath(os.path.join(data_dir, filename))

    if not (requested == data_dir or requested.startswith(data_dir + os.sep)):
        return "Forbidden", 403

    if not os.path.exists(requested):
        return "Not found", 404

    return send_from_directory(data_dir, filename)


@app.route("/user-inference/<path:filename>")
@login_required
def user_inference_files(filename):
    user_slug = get_current_user_slug()
    data_dir = os.path.abspath(os.path.join(user_root(user_slug), "inference"))
    requested = os.path.abspath(os.path.join(data_dir, filename))

    if not (requested == data_dir or requested.startswith(data_dir + os.sep)):
        return "Forbidden", 403

    if not os.path.exists(requested):
        return "Not found", 404

    return send_from_directory(data_dir, filename)



@app.route('/')
@login_required
def index():
    return render_template('index.html', is_homepage=True)



@app.route('/dataset')
@login_required
def dataset():
    # Page Logic
    name       = request.args.get('id')
    mode       = request.args.get("mode", default="segmentation", type=str)
    page       = request.args.get("page", default=1, type=int)
    pager_size = request.args.get(
        "pager_size",
        default=config.get("defaults").get("pager_size"),
        type=int
    )

    if not name:
        return "Dataset not found", 404
    
    user_slug = get_current_user_slug()
    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")

    dataset_path = {
        "segmentation"  : f"{user_datasets_root}/Segmentation/",
        "detection"     : f"{user_datasets_root}/Object-Detection/",
        "classification": f"{user_datasets_root}/Classification/"
    }

    dataset_info, dataset_items = load_dataset_info(
        filepath = dataset_path[mode],
        name     = name,
        mode     = mode
    )

    max_w, max_h = get_max_image_size(dataset_items)
    dataset_info["max_width"]  = max_w
    dataset_info["max_height"] = max_h

    for item in dataset_items:
        image_path = item.get("image")
        if image_path:
            rel_path = os.path.relpath(image_path, user_datasets_root)
            item["image_url"] = url_for("user_dataset_files", filename=rel_path)
            item["display_name"] = os.path.basename(image_path)

    pager_size_options = [15, 30, 50, 80, 120]
    # Don't overload the page
    visible_items = dataset_items[pager_size*(page-1):pager_size*page-1]
    num_pages = math.ceil(len(dataset_items) / pager_size)

    return render_template(
        "dataset.html",
        mode               = mode,
        dataset_info       = dataset_info,
        dataset_items      = visible_items,
        page               = page,
        pager_size         = pager_size,
        pager_size_options = pager_size_options,
        num_pages          = num_pages,
    )



# Datasets page showing all available datasets with metadata
@app.route('/collections')
@login_required
def collections():
    # Dataset logic
    user_slug = get_current_user_slug()
    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")

    seg_datasets = load_datasets(f"{user_datasets_root}/Segmentation", "Seg")
    od_datasets  = load_datasets(f"{user_datasets_root}/Object-Detection", "OD")
    cls_datasets = load_datasets(f"{user_datasets_root}/Classification", "Cls")

    def enrich(datasets, base_path):
        enriched = []
        for d in datasets:
            folder_path = os.path.join(base_path, d["id"])
            creation_timestamp = get_best_timestamp(folder_path)
            creation_date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y")
            enriched.append({
                "name": d["name"],
                "num_samples": d["num_samples"],
                "type": "2D images",
                "date": creation_date,
                "url": "/dataset"
            })
        return enriched

    seg_datasets = enrich(seg_datasets, f"{user_datasets_root}/Segmentation")
    od_datasets  = enrich(od_datasets, f"{user_datasets_root}/Object-Detection")
    cls_datasets = enrich(cls_datasets, f"{user_datasets_root}/Classification")

    datasets = {
        "segmentation"  : seg_datasets,
        "detection"     : od_datasets,
        "classification": cls_datasets
    }

    # Page settings
    mode     = request.args.get("mode")
    msg      = request.args.get("msg")
    msg_type = request.args.get("msg_type")
    
    if mode is None:
        mode = "segmentation"

    if msg is not None: 
        flash(msg, msg_type)
    
    return render_template(
        "collections.html",
        mode=mode,
        datasets=datasets[mode]
    )


@app.route('/add_dataset')
@login_required
def add_dataset():
    num_classes = {
        "default": 2,
        "min"    : 1,
        "max"    : 64
    }

    # Page Logic
    mode = request.args.get("mode")

    if mode is None:
        mode = "segmentation"

    return render_template(
        'add_dataset.html',
        mode=mode,
        num_classes=num_classes
    )


@app.route('/upload_dataset_zip', methods=['POST'])
@login_required
def upload_dataset_zip():

    file = request.files.get('dataset_zip')
    if not file or file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'status': 'error', 'message': 'Only .zip files allowed'}), 400
    
    filename = secure_filename(file.filename)
    user_slug = get_current_user_slug()
    save_dir = os.path.join(user_root(user_slug), "tmp_datasets_zips")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    try:
        file.save(path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save file: {e}'}), 500
    return jsonify({'status': 'ok', 'filename': filename})


def process_dataset(mode, zip_path, num_classes=None, user_slug=None):
    if user_slug is None:
        user_slug = get_current_user_slug()

    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")
    DEST_PATHS = {
        "segmentation"  : f"{user_datasets_root}/Segmentation",
        "detection"     : f"{user_datasets_root}/Object-Detection",
        "classification": f"{user_datasets_root}/Classification"
    }

    # unzip to a temp folder
    tmp_root = os.path.join(user_root(user_slug), "tmp_datasets")
    os.makedirs(tmp_root, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    tmp_dir = os.path.join(tmp_root, dataset_name)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp_dir)
    except Exception as e:
        return False, f"Failed to unzip: {e}"

    # If the zip contains a single folder, use that as root
    ins_folder = os.listdir(tmp_dir)
    if len(ins_folder) == 1:
        candidate = os.path.join(tmp_dir, ins_folder[0])
        if os.path.isdir(candidate):
            tmp_dir = candidate

    def same_files_no_ext(dir1, dir2):
        f1 = {os.path.splitext(f)[0] for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
        f2 = {os.path.splitext(f)[0] for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
        return f1 == f2, f1, f2

    # Verify rules to accept the dataset
    if mode == "segmentation":
        req = [
            "images/train",
            "images/val",
            "masks/train",
            "masks/val",
            "labelmap.txt"
        ]
        for r in req:
            if not os.path.exists(os.path.join(tmp_dir, r)):
                shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                return False, f"Missing: {r}"
        
        # Check that images/train and masks/train match
        img_train = os.path.join(tmp_dir, "images/train")
        msk_train = os.path.join(tmp_dir, "masks/train")

        ok, img_set, msk_set = same_files_no_ext(img_train, msk_train)
        if not ok:
            missing_masks = img_set - msk_set
            missing_images = msk_set - img_set
            return False, f"Train mismatch. Images missing masks: {missing_masks}, masks missing images: {missing_images}"

        # Check that images/val and masks/val match
        img_val = os.path.join(tmp_dir, "images/val")
        msk_val = os.path.join(tmp_dir, "masks/val")

        ok, img_set, msk_set = same_files_no_ext(img_val, msk_val)
        if not ok:
            missing_masks = img_set - msk_set
            missing_images = msk_set - img_set
            return False, f"Val mismatch. Images missing masks: {missing_masks}, masks missing images: {missing_images}"

    elif mode == "detection":
        if not os.path.isfile(os.path.join(tmp_dir, "data.yaml")):
            shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
            return False, "Missing data.yaml"
        
        # Check OD train split
        train_img = os.path.join(tmp_dir, "train/images")
        train_lbl = os.path.join(tmp_dir, "train/labels")

        if not os.path.exists(train_img) or not os.path.exists(train_lbl):
            return False, "Invalid YOLO structure: missing train/images or train/labels"

        ok, img_set, lbl_set = same_files_no_ext(train_img, train_lbl)
        if not ok:
            missing_labels = img_set - lbl_set
            missing_images = lbl_set - img_set
            return False, f"Train mismatch. Images missing labels: {missing_labels}, labels missing images: {missing_images}"

        # Check OD val split
        val_img = os.path.join(tmp_dir, "valid/images")
        val_lbl = os.path.join(tmp_dir, "valid/labels")

        if not os.path.exists(val_img) or not os.path.exists(val_lbl):
            return False, "Invalid YOLO structure: missing valid/images or valid/labels"

        ok, img_set, lbl_set = same_files_no_ext(val_img, val_lbl)
        if not ok:
            missing_labels = img_set - lbl_set
            missing_images = lbl_set - img_set
            return False, f"Val mismatch. Images missing labels: {missing_labels}, labels missing images: {missing_images}"

    elif mode == "classification":
        def validate_class_dir(root_dir, split_label):
            if not os.path.isdir(root_dir):
                return False, f"Missing '{split_label}' folder"

            subdirs = [
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]

            if len(subdirs) != int(num_classes):
                return False, f"Expected {num_classes} class folders in '{split_label}', found {len(subdirs)}"

            for cls in subdirs:
                cls_path = os.path.join(root_dir, cls)
                contents = os.listdir(cls_path)

                # Must NOT be empty
                if len(contents) == 0:
                    return False, f"Class '{cls}' in '{split_label}' is empty"

                # Must NOT contain directories
                for item in contents:
                    if os.path.isdir(os.path.join(cls_path, item)):
                        return False, (
                            f"Class '{cls}' in '{split_label}' contains a folder ('{item}') "
                            "but only image files are allowed"
                        )

            return True, set(subdirs)

        train_dir = os.path.join(tmp_dir, "train")
        val_dir = os.path.join(tmp_dir, "val")
        dataset_already_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)

        if dataset_already_split:
            ok, train_classes = validate_class_dir(train_dir, "train")
            if not ok:
                shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                return False, train_classes

            ok, val_classes = validate_class_dir(val_dir, "val")
            if not ok:
                shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                return False, val_classes

            if train_classes != val_classes:
                shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                missing_train = val_classes - train_classes
                missing_val = train_classes - val_classes
                return False, (
                    "Train/val class mismatch. "
                    f"Missing in train: {missing_train}, missing in val: {missing_val}"
                )
        else:
            subdirs = [
                d for d in os.listdir(tmp_dir)
                if os.path.isdir(os.path.join(tmp_dir, d))
            ]
            if len(subdirs) != int(num_classes):
                shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                return False, f"Expected {num_classes} class folders, found {len(subdirs)}"
    
            for cls in subdirs:
                cls_path = os.path.join(tmp_dir, cls)
                contents = os.listdir(cls_path)

                # Must NOT be empty
                if len(contents) == 0:
                    shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                    return False, f"Class '{cls}' is empty"

                # Must NOT contain directories
                for item in contents:
                    if os.path.isdir(os.path.join(cls_path, item)):
                        shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
                        return False, f"Class '{cls}' contains a folder ('{item}') but only image files are allowed"

    else:
        return False, "Unknown mode"

    # Move to the Datasets Folder
    final_root = DEST_PATHS[mode]
    os.makedirs(final_root, exist_ok=True)

    final_path = os.path.join(final_root, os.path.basename(tmp_dir))
    if os.path.exists(final_path):
        shutil.rmtree(final_path)

    shutil.move(tmp_dir, final_path)

    # Clean the temp folder and zip
    shutil.rmtree(os.path.dirname(tmp_dir), ignore_errors=True)
    os.remove(zip_path)

    return True, "Dataset imported successfully"


@app.route('/dataset_submit', methods=["POST"])
@login_required
def dataset_submit():
    mode        = request.form.get('mode')
    num_classes = request.form.get('num_classes')
    filename    = request.form.get('dataset_zip')

    user_slug = get_current_user_slug()
    zip_path = os.path.join(user_root(user_slug), "tmp_datasets_zips", filename)

    if not os.path.isfile(zip_path):
        error_msg = "Uploaded dataset zip not found"
        flash(error_msg, "danger")
        return redirect(url_for("collections", mode=mode, msg=error_msg, msg_type="danger"))

    success, msg = process_dataset(mode, zip_path, num_classes, user_slug=user_slug)

    if not success:
        return redirect(url_for("collections", mode=mode, msg=msg, msg_type="danger"))

    return redirect(url_for("collections", mode=mode, msg=msg, msg_type="info"))

@app.route('/train_model', methods=['GET'])
@login_required
def train_model():
    # Models
    seg_models = load_models_available('static/models_available/models_available_segmentation.csv')
    cls_models = load_models_available('static/models_available/models_available_classification.csv')
    od_models  = load_models_available('static/models_available/models_available_object_detection.csv')

    models = {
        "segmentation"  : seg_models,
        "detection"     : od_models,
        "classification": cls_models
    }

    # Collections
    user_slug = get_current_user_slug()
    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")

    od_collections  = load_datasets(f"{user_datasets_root}/Object-Detection","OD")
    seg_collections = load_datasets(f"{user_datasets_root}/Segmentation","Seg")
    cls_collections = load_datasets(f"{user_datasets_root}/Classification","Cls")

    collections = {
        "segmentation"  : seg_collections,
        "detection"     : od_collections,
        "classification": cls_collections
    }

    # Advanced
    advanced_config = {
        "learning_rate": {
            "range"    : "discrete",
            "values"   : [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "lower"    : 0.00001,
            "upper"    : 0.1,
            "lower_def": config.get('defaults').get('lr_lower'),
            "upper_def": config.get('defaults').get('lr_upper'),
            "description": "Controls the step size for model weight updates during training. Higher values train faster but may miss optimal solutions."
        },
        "batch_size": {
            "range"    : "discrete",
            "values"   : [1, 2, 4, 8, 16, 32, 64],
            "lower"    : 1,
            "upper"    : 64,
            "lower_def": config.get('defaults').get('bs_lower'),
            "upper_def": config.get('defaults').get('bs_upper'),
            "description": "Number of samples processed in each training iteration. Larger batches are more stable but require more memory."
        },
        "epochs": {
            "range"    : "continuous",
            "min"      : 1,
            "max"      : 100,
            "step"     : 1,
            "lower"    : 1,
            "upper"    : 100,
            "lower_def": config.get('defaults').get('ep_lower'),
            "upper_def": config.get('defaults').get('ep_upper'),
            "description": "Number of complete passes through the training dataset. More epochs allow deeper learning but risk overfitting."
        },
    }

    augmentations = {
        "segmentation": {
            "blur"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('seg_blur'),
                "description": "Applies Gaussian blur to simulate motion or focus blur in images."
            },
            "scale" : {
                "type": "bool",
                "default": config.get('defaults').get('seg_scale'),
                "description": "Randomly scales images to different sizes for improved robustness."
            },
            "rotate": {
                "type": "bool",
                "default": config.get('defaults').get('seg_rotate'),
                "description": "Randomly rotates images to improve generalization to different orientations."
            },
            "flip"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('seg_flip'),
                "description": "Flips images horizontally or vertically for spatial invariance."
            }
        },
        "classification": {
            "blur"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_blur'),
                "description": "Applies Gaussian blur to simulate motion or focus blur in images."
            },
            "rotate": {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_rotate'),
                "description": "Randomly rotates images to improve generalization to different orientations."
            },
            "flip"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_flip'),
                "description": "Flips images horizontally or vertically for spatial invariance."
            }
        },
        "detection": {
            "hue": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
                "max_val_default": config.get('defaults').get('det_hue'),
                "unit"           : None,
                "description": "Adjusts color hue to simulate different lighting conditions and improve color robustness."
            },
            "saturation": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "max_val_default": config.get('defaults').get('det_sat'),
                "unit"           : None,
                "description": "Modifies color saturation to improve the model's invariance to color variations."
            },
            "value": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "max_val_default": config.get('defaults').get('det_val'),
                "unit"           : None,
                "description": "Changes brightness values to handle varying illumination and improve brightness robustness."
            },
            "flip": {
                "type"           : "range",
                "default"        : True,
                "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "prob_default"   : config.get('defaults').get('det_flip'),
                "unit"           : None,
                "description": "Randomly flips images horizontally with specified probability for spatial invariance."
            },
            "rotate": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0, 30, 60, 90, 120, 150, 180],
                "max_val_default": config.get('defaults').get('det_rotate'),
                "unit"           : "degrees",
                "description": "Randomly rotates images within specified angle range to improve rotational invariance."
            },
        }
    }

    # Page settings
    mode = request.args.get("mode")

    if mode is None:
        mode = "segmentation"
    
    return render_template(
        'train_model.html',
        mode            = mode,
        models          = models[mode],
        collections     = collections[mode],
        advanced_config = advanced_config,
        augmentations   = augmentations[mode]
    )


# Page for the trained models
@app.route('/models')
@login_required
def models():
    user_slug = get_current_user_slug()
    ensure_user_folders(user_slug)

    seg_models = load_models(os.path.join(user_root(user_slug), "models_db", "trained_models_db_segm.csv"))
    od_models  = load_models(os.path.join(user_root(user_slug), "models_db", "trained_models_db_od.csv"))
    cls_models = load_models(os.path.join(user_root(user_slug), "models_db", "trained_models_db_cls.csv"))

    models = {
        "segmentation"  : seg_models,
        "detection"     : od_models,
        "classification": cls_models
    }

    # Page settings
    mode     = request.args.get("mode")
    msg      = request.args.get("msg")
    msg_type = request.args.get("msg_type")
    
    if mode is None:
        mode = "segmentation"

    if msg is not None: 
        flash(msg, msg_type)
    return render_template(
        'models.html',
        mode    = mode,
        models  = models[mode],
    )


@app.route('/train_model_submit', methods=["POST"])
@login_required
def train_model_submit():
    user_slug = get_current_user_slug()
    ensure_user_folders(user_slug)
    mode = request.form.get('mode')

    selected_model      = request.form.get('model')
    selected_collection = request.form.get('collection')

    # Advanced
    lr_left         = request.form.get('learning_rate_left')
    lr_right        = request.form.get('learning_rate_right')
    bs_left         = request.form.get('batch_size_left')
    bs_right        = request.form.get('batch_size_right')
    epoch_left      = request.form.get('epochs_left')
    epoch_right     = request.form.get('epochs_right')

    # Augmentations
    blur_enabled    = request.form.get('blur_enabled', 'false')
    scale_enabled   = request.form.get('scale_enabled', 'false')
    rotate_enabled  = request.form.get('rotate_enabled', 'false')
    flip_enabled    = request.form.get('flip_enabled', 'false')

    hue_enabled = request.form.get('hue_enabled', 'false')
    hue_prob    = request.form.get('hue_prob', 0.0)
    hue_maxval  = request.form.get('hue_maxval', 0.0)
    
    sat_enabled = request.form.get('saturation_enabled', 'false')
    sat_prob    = request.form.get('saturation_prob', 0.0)
    sat_maxval  = request.form.get('saturation_maxval', 0.0)

    val_enabled = request.form.get('value_enabled', 'false')
    val_prob    = request.form.get('value_prob', 0.0)
    val_maxval  = request.form.get('value_maxval', 0.0)

    rot_enabled = request.form.get('rotate_enabled', 'false')
    rot_prob    = request.form.get('rotate_prob', 0.0)
    rot_maxval  = request.form.get('rotate_maxval', 0.0)

    flp_enabled = request.form.get('flip_enabled', 'false')
    flp_prob    = request.form.get('flip_prob', 0.0)

    print(selected_collection)
    print(selected_model)
    print("Lr Left:", lr_left)
    print("Lr Right:", lr_right)
    print("Batch Size Left:", bs_left)
    print("Batch Size Right:", bs_right)
    print("Epochs Left:", epoch_left)
    print("Epochs Right:", epoch_right)

    def bool_str(v):
        return "true" if str(v).lower() == "true" else "false"

    base_args = [
        "--model", selected_model,
        "--dataset", selected_collection,
        "--lr_left", lr_left,
        "--lr_right", lr_right,
        "--bs_left", bs_left,
        "--bs_right", bs_right,
        "--epoch_left", epoch_left,
        "--epoch_right", epoch_right,
    ]

    # Segmentation command
    seg_cmd = [
        "python", "training_button_segmentation.py",
        "--user", user_slug,
    ] + base_args + [
        "--blur", bool_str(blur_enabled),
        "--scale", bool_str(scale_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
    ]

    # Classification command
    cls_split_flag = False
    if mode == "classification" and selected_collection:
        dataset_root = os.path.join(user_root(user_slug), "Datasets", "Classification", selected_collection)
        train_dir = os.path.join(dataset_root, "train")
        val_dir = os.path.join(dataset_root, "val")
        cls_split_flag = os.path.isdir(train_dir) and os.path.isdir(val_dir)

    cls_cmd = [
        "python", "training_button_classification.py",
        "--user", user_slug,
    ] + base_args + [
        "--blur", bool_str(blur_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
        "--dataset_already_split", bool_str(cls_split_flag),
    ]

    # Detection command
    det_cmd = [
        "python", "training_button_object_detection.py",
        "--user", user_slug,
    ] + base_args

    if bool_str(hue_enabled) == "true":
        det_cmd += ["--hue", hue_maxval]
    else: 
        det_cmd += ["--hue", "0.0"]

    if bool_str(sat_enabled) == "true":
        det_cmd += ["--saturation", sat_maxval]
    else: 
        det_cmd += ["--saturation", "0.0"]
        
    if bool_str(val_enabled) == "true":
        det_cmd += ["--value", val_maxval]
    else: 
        det_cmd += ["--value", "0.0"]

    if bool_str(rot_enabled) == "true":
        det_cmd += ["--rotate", rot_maxval]
    else: 
        det_cmd += ["--rotate", "0.0"]

    if bool_str(flp_enabled) == "true":
        det_cmd += ["--flip", flp_prob]
    else: 
        det_cmd += ["--flip", "0.0"]

    subprocesses = {
        "segmentation"  : seg_cmd,
        "detection"     : det_cmd,
        "classification": cls_cmd
    }

    paths = {
        "segmentation" : [
            os.path.join(user_root(user_slug), "Segmentation", "runs", "user_experiments.csv"),
            os.path.join(user_root(user_slug), "models_db", "trained_models_db_segm.csv"),
            "Seg",
            user_slug,
        ],
        "detection" : [
            os.path.join(user_root(user_slug), "ObjectDetection", "runs", "user_experiments.csv"),
            os.path.join(user_root(user_slug), "models_db", "trained_models_db_od.csv"),
            "OD",
            user_slug,
        ],
        "classification": [
            os.path.join(user_root(user_slug), "Classification", "runs", "user_experiments.csv"),
            os.path.join(user_root(user_slug), "models_db", "trained_models_db_cls.csv"),
            "Cls",
            user_slug,
        ]
    }
    
    job_id = str(uuid.uuid4())
    _write_job_status(user_slug, job_id, {"status": "running", "mode": mode})

    thread = threading.Thread(
        target=_run_training_job,
        args=(user_slug, job_id, subprocesses[mode], mode, paths),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "running", "job_id": job_id})


@app.route('/train_status/<job_id>', methods=["GET"])
@login_required
def train_status(job_id):
    user_slug = get_current_user_slug()
    status = _read_job_status(user_slug, job_id)
    if not status:
        return jsonify({"error": "job_not_found"}), 404
    return jsonify(status)


@app.route('/inference', methods=['GET', 'POST'])
@login_required
def inference():
    model_id = int(request.args.get("id"))
    mode     = request.args.get("mode")
    user_slug = get_current_user_slug()
    ensure_user_folders(user_slug)

    # Init params
    params = {
        "segmentation": {
            "csv"   : os.path.join(user_root(user_slug), "models_db", "trained_models_db_segm.csv"),
            "metric": "mIoU Score"
        },
        "detection": {
            "csv"   : os.path.join(user_root(user_slug), "models_db", "trained_models_db_od.csv"),
            "metric": "mAP 50-95 Score"
        },
        "classification": {
            "csv"   : os.path.join(user_root(user_slug), "models_db", "trained_models_db_cls.csv"),
            "metric": "Accuracy"
        },
    
    }
    mode_params = params[mode]
    model_csv   = mode_params["csv"]
    metric      = mode_params["metric"]

    with open(model_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        models = [row for row in reader]
    
    # Match ui input with the list of models
    model = next((m for i, m in enumerate(models, 1) if i == model_id), None)
    if not model:
        return "Model not found", 404
    
    error_msg   = None
    success_msg = None

    results         = []
    model_name      = model['name']
    checkpoint_path = model['checkpoint_path']
    config_path     = model['config_path']
    color_table     = []                        # for color map (segmentation mode)
    dataset_name    = model['trained_on']
    inference_root = os.path.join(user_root(user_slug), "inference", mode)
    
    if request.method == 'POST':
        if mode == "segmentation":
            files = request.files.getlist('image')  # Get multiple files
            if files:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                save_dir = os.path.join(inference_root, "inputs", str(model_id), timestamp)
                output_dir = os.path.join(inference_root, "outputs", str(model_id), timestamp)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save all uploaded files
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(save_dir, filename)
                        file.save(filepath)
                
                # Temporary container creation
                container = client.containers.run(
                    IMAGE_SEGM_CLS,
                    command="sleep infinity",
                    detach=True,
                    volumes={BASE_HOST_PATH_OUT: {"bind": "/data", "mode": "rw"}}
                )
                try:
                    container.exec_run([
                        "python", "/data/Segmentation/inference_cvat.py",
                        "-c", config_path,
                        "-m", checkpoint_path,
                        "-u", user_slug,
                        "-i", f"/data/{user_slug}/inference/segmentation/inputs/{model_id}/{timestamp}/",
                        "-o", f"/data/{user_slug}/inference/segmentation/outputs/{model_id}/{timestamp}/"
                    ])

                    container.exec_run([
                        "chown", "-R", "1003:1003", f"/data/{user_slug}/inference/segmentation/outputs/"
                    ])

                finally:
                    container.stop()
                    container.remove()

                # Get all output files
                output_data_dir = output_dir
                output_files    = glob.glob(os.path.join(output_data_dir, '*.png')) + glob.glob(os.path.join(output_data_dir, '*.jpg'))
                
                # Create a mapping of output files by name
                output_map = {os.path.basename(f): f for f in output_files}
                
                # Process each uploaded file and match with output
                for file in files:
                    if file.filename:
                        filename   = secure_filename(file.filename)
                        input_image = url_for("user_inference_files", filename=f"segmentation/inputs/{model_id}/{timestamp}/{filename}")
                        
                        # Find matching output file
                        output_image = None
                        base_name    = os.path.splitext(filename)[0]
                        for output_name, output_path in output_map.items():
                            if base_name in output_name:
                                output_image = url_for("user_inference_files", filename=f"segmentation/outputs/{model_id}/{timestamp}/{output_name}")
                                break
                        
                        results.append({
                            'input_image' : input_image,
                            'output_image': output_image
                        })
            
            # Load datasets coloring to propagate it to the inference page
            if dataset_name:
                dataset_file = os.path.join(user_root(user_slug), "Datasets", "Segmentation", dataset_name, "labelmap.txt")

                if os.path.exists(dataset_file):
                    with open(dataset_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue

                            # Format: label:color_rgb:parts:actions
                            parts = line.split(":")
                            if len(parts) >= 2:
                                label = parts[0]
                                rgb   = parts[1]
                                r, g, b = map(int, rgb.split(","))

                                color_table.append({
                                    "label": label,
                                    "rgb": f"rgb({r},{g},{b})",
                                    "hex": "#{:02x}{:02x}{:02x}".format(r, g, b)
                                })
                
                success_msg = f"Inference for {len(files)} image(s) completed!"
        
        if mode == "detection":
            files = request.files.getlist('image')  # Get multiple files
            if files:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                save_dir = os.path.join(inference_root, "inputs", str(model_id), timestamp)
                output_dir = os.path.join(inference_root, "outputs", str(model_id), timestamp)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save all uploaded files
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(save_dir, filename)
                        file.save(filepath)

                # Temporary container creation
                container = client.containers.run(
                    IMAGE_OD,             
                    command="sleep infinity",
                    detach=True,
                    volumes={BASE_HOST_PATH_OUT: {"bind": "/data", "mode": "rw"}}
                )

                try:
                    # Run inference
                    container.exec_run([
                        "python", "/data/ObjectDetection/ultralytics/inference_v1.py",
                        "--checkpoint", checkpoint_path,
                        "--input", f"/data/{user_slug}/inference/detection/inputs/{model_id}/{timestamp}/",
                        "--output", f"/data/{user_slug}/inference/detection/outputs/{model_id}/{timestamp}/",
                    ])

                    # ownership fix
                    container.exec_run([
                        "chown", "-R", "1003:1003", "/data/"
                    ])

                finally:
                    container.stop()
                    container.remove()


                # Collect outputs from predict directory
                output_data_dir = os.path.join(output_dir, "predict")
                output_files    = glob.glob(os.path.join(output_data_dir, '*.png')) + glob.glob(os.path.join(output_data_dir, '*.jpg'))
                
                # Create a mapping of output files by name (without extension)
                output_map = {}
                for f in output_files: 
                    base = os.path.splitext(os.path.basename(f))[0]
                    output_map[base] = f
                
                print(f"Output map: {output_map}")
                
                # Process each uploaded file and match with output
                for file in files:
                    if file.filename:
                        filename   = secure_filename(file.filename)
                        input_image = url_for("user_inference_files", filename=f"detection/inputs/{model_id}/{timestamp}/{filename}")
                        
                        # Find matching output file
                        output_image = None
                        base_name    = os.path.splitext(filename)[0]
                        
                        if base_name in output_map:
                            output_path     = output_map[base_name]
                            output_filename = os.path.basename(output_path)
                            output_image = url_for("user_inference_files", filename=f"detection/outputs/{model_id}/{timestamp}/predict/{output_filename}")
                        else:
                            print(f"WARNING: No output found for {filename}")
                        
                        results.append({
                            'input_image' : input_image,
                            'output_image': output_image
                        })
                
                success_msg = f"Inference for {len(files)} image(s) completed!"

        if mode == "classification":
            files = request.files.getlist('file')  # Get multiple files
            if files:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                save_dir = os.path.join(inference_root, "inputs", str(model_id), timestamp)
                output_dir = os.path.join(inference_root, "outputs", str(model_id), timestamp)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                # Process each file individually
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(save_dir, filename)
                        file.save(filepath)

                    # Temporary container creation
                    container = client.containers.run(
                        IMAGE_SEGM_CLS,
                        command="sleep infinity",
                        detach=True,
                        volumes={BASE_HOST_PATH_OUT: {"bind": "/data", "mode": "rw"}}
                    )

                    try:
                        # Run inference
                        container.exec_run([
                            "python", "/data/Classification/inference.py",
                            "--model_path", checkpoint_path,
                            "--model", model_name,
                            "--class_names", config_path,
                            "--image", f"/data/{user_slug}/inference/classification/inputs/{model_id}/{timestamp}/{filename}",
                            "--output_dir", f"/data/{user_slug}/inference/classification/outputs/{model_id}/{timestamp}/"
                        ])

                        # Fix permissions
                        container.exec_run([
                            "chown", "-R", "1003:1003", f"/data/{user_slug}/inference/classification/outputs/"
                        ])

                    finally:
                        container.stop()
                        container.remove()

                    input_file = url_for("user_inference_files", filename=f"classification/inputs/{model_id}/{timestamp}/{filename}")

                    # Find the corresponding output file by matching the base name
                    output_data_dir  = output_dir
                    base_name        = os.path.splitext(filename)[0]  # Get filename without extension
                    expected_output  = f"{base_name}.txt"
                    output_file_path = os.path.join(output_data_dir, expected_output)
                    
                    output_text = None
                    if os.path.exists(output_file_path):
                        # Read the output text
                        with open(output_file_path, 'r', encoding='utf-8') as f:
                            output_text = f.read()
                    else:
                        print(f"WARNING: Output file not found for {filename}: {output_file_path}")
                    
                    results.append({
                        'input_file' : input_file,
                        'output_text': output_text,
                        'filename'   : filename
                    })

            success_msg = f"Inference for {len(files)} image(s) completed!"
    
    if success_msg: 
        flash(success_msg, "info")
    if error_msg:
        flash(error_msg, "danger")
    
    return render_template(
        'inference.html',
        mode         = mode,
        model_id     = model_id,
        model        = model,
        results      = results,
        metric_label = metric,
        color_table  = color_table,
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8056)