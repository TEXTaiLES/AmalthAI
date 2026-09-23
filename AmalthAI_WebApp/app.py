from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, send_from_directory, g, session
from flask_login import login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import tomllib
import math
import os
import subprocess
import time
import glob
import csv
import shutil
import threading
import json
import uuid
import docker
import requests
import base64
import mimetypes
from io import BytesIO
from urllib.parse import quote
from datetime import datetime
from utils.auth import init_auth
from utils.models_page import write_results
from utils.helpers import create_multispectral_preview, load_datasets, load_models_available, load_dataset_info, get_max_image_size, load_models, get_best_timestamp
from utils.job_status import _job_status_path, _mark_stale_if_dead, _write_job_status, _read_job_status, _run_training_job
from utils import user_paths
from utils.user_paths import safe_user_slug, user_root, get_current_user_slug, get_current_user_email, ensure_user_folders
from utils import dataset_processing
from utils.dataset_processing import process_dataset, _merged_datasets
from utils.vlm_utils import build_user_prompt
from utils.vlm_utils import SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_BLANKS, USER_PROMPT_TEMPLATE, USER_PROMPT_FIELDS, USER_PROMPT_CLASS_EXAMPLES
from utils import hestia_helpers
from utils import inference_helpers
from utils.inference_helpers import _inference_handoff_path, _parse_classification_output
from utils.inference_helpers import _collect_available_inference_runs, _has_available_inference_results, _inference_params, _load_segmentation_color_table, _resolve_inference_model, _store_vlm_handoff
from utils.hestia_helpers import _dataset_manifest, _persist_dataset_to_hestia, _push_trained_model_to_hestia, _hestia_models_for_template, _persist_inference_to_hestia
from utils.load_config import load_config, chown_target
from utils import hestia_client as hc
from utils.zip_utils import safe_extract_zip

# Yaml config
config = load_config("config.yml")

# Connection with docker containers
client = docker.from_env()

# App initialization
app = Flask(__name__)
app.config["SECRET_KEY"] = config["flask"]["secret_key"]
app.config.from_file('config.toml', load=tomllib.load, text=False)

# Login and logout routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    next_path = request.args.get("next") or url_for("index")
    target = request.host_url.rstrip("/") + next_path
    return redirect(f"{DIRECTUS_LOGIN_URL}?redirect_url={quote(target, safe='')}")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    response = redirect(url_for("login"))
    response.delete_cookie(SHARED_REFRESH_COOKIE_NAME, path="/", domain=_auth_cookie_domain())
    return response

# App configurations
DIRECTUS_BASE_URL = config.get("directus", {}).get("base_url")
DIRECTUS_LOGIN_URL = config.get("directus", {}).get("login_url")
SHARED_REFRESH_COOKIE_NAME = config.get("textailes-token", {}).get("token")

BASE_HOST_PATH = config.get("paths").get("base_host_path")
BASE_HOST_PATH_OUT = config.get("paths").get("base_host_path_out")
user_paths.init(BASE_HOST_PATH)
IMAGE_SEGM_CLS = config.get("images").get("classification")
IMAGE_OD = config.get("images").get("detection")

# HESTIA data-lake integration
HESTIA_ENABLED = bool(config.get("hestia", {}).get("enabled"))
hc.configure(base_url=config.get("hestia", {}).get("base_url"))

# VLM integration
VLM_URL = config.get("vlm", {}).get("base_url")

# Per-mode constants used by the HESTIA hooks.
HESTIA_METRIC = {
    "segmentation": "mIoU Score",
    "detection": "mAP 50-95 Score",
    "classification": "Accuracy",
    "multispectral_classification": "Accuracy",
}

# AmalthAI dataset directories.
HESTIA_DATASET_DIR = {
    "segmentation": "Segmentation",
    "detection": "Object-Detection",
    "classification": "Classification",
    "multispectral_classification": "Multispectral-Classification",
}

dataset_processing.init(HESTIA_ENABLED, HESTIA_DATASET_DIR)
hestia_helpers.init(HESTIA_METRIC)
inference_helpers.init(HESTIA_ENABLED)

auth_helpers = init_auth(
    app,
    directus_base_url=DIRECTUS_BASE_URL,
    refresh_cookie_name=SHARED_REFRESH_COOKIE_NAME,
    safe_user_slug=safe_user_slug,
    ensure_user_folders=ensure_user_folders,
)

_register_user = auth_helpers["register_user"]
_store_shared_auth = auth_helpers["store_shared_auth"]
_cached_access_token = auth_helpers["cached_access_token"]
_fetch_current_identity = auth_helpers["fetch_current_identity"]
_auth_cookie_domain = auth_helpers["auth_cookie_domain"]
_auth_cookie_secure = auth_helpers["auth_cookie_secure"]
_directus_payload = auth_helpers["directus_payload"]

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


@app.route("/multispectral-preview/<path:filename>")
@login_required
def multispectral_preview(filename):
    user_slug = get_current_user_slug()
    data_dir = os.path.abspath(os.path.join(user_root(user_slug), "Datasets"))
    requested = os.path.abspath(os.path.join(data_dir, filename))

    if not requested.startswith(data_dir + os.sep):
        return "Forbidden", 403

    parts = os.path.normpath(filename).split(os.sep)
    if len(parts) < 3 or parts[0] != HESTIA_DATASET_DIR["multispectral_classification"]:
        return "Invalid multispectral path", 400

    metadata_path = os.path.join(data_dir, parts[0], parts[1], "multispectral_metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        num_channels = int(metadata.get("num_channels") or metadata.get("in_channels"))
        image = create_multispectral_preview(requested, num_channels)
        output = BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png")
    except (OSError, TypeError, ValueError, KeyError):
        return "Preview unavailable", 400


@app.route("/multispectral-inference-preview/<path:filename>")
@login_required
def multispectral_inference_preview(filename):
    user_slug = get_current_user_slug()
    data_dir = os.path.realpath(os.path.join(user_root(user_slug), "inference"))
    requested = os.path.realpath(os.path.join(data_dir, filename))
    parts = os.path.normpath(filename).split(os.sep)
    prefix = parts[:2] == ["multispectral_classification", "inputs"]
    hestia_prefix = parts[:3] == ["_hestia_results", "multispectral_classification", "inputs"]

    if not requested.startswith(data_dir + os.sep):
        return "Forbidden", 403
    if not (prefix or hestia_prefix) or os.path.splitext(filename)[1].lower() not in (".tif", ".tiff"):
        return "Invalid multispectral path", 400

    try:
        image = create_multispectral_preview(requested)
        output = BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png")
    except (OSError, TypeError, ValueError, KeyError):
        return "Preview unavailable", 400


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
    page       = request.args.get("page", default=1, type=int) or 1
    pager_size = request.args.get("pager_size", default=config.get("defaults").get("pager_size"), type=int)
    page       = max(page, 1)

    if not pager_size or pager_size < 1:
        pager_size = config.get("defaults").get("pager_size")

    if not name:
        return "Dataset not found", 404
    
    user_slug = get_current_user_slug()
    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")

    dataset_path = {
        "segmentation"  : f"{user_datasets_root}/Segmentation/",
        "detection"     : f"{user_datasets_root}/Object-Detection/",
        "classification": f"{user_datasets_root}/Classification/",
        "multispectral_classification": f"{user_datasets_root}/Multispectral-Classification/"
    }

    if mode not in dataset_path:
        return "Invalid mode", 400

    # HESTIA: rehydrate the dataset into the local cache so a dataset that lives
    # only in the data lake (not cached on this node) can still be viewed.
    if HESTIA_ENABLED:
        hc.ensure_dataset_local(user_slug, mode, name, dataset_path[mode])

    dataset_info, dataset_items = load_dataset_info(
        filepath = dataset_path[mode],
        name     = name,
        mode     = mode
    )

    max_w, max_h = get_max_image_size(dataset_items)
    dataset_info["max_width"]  = max_w
    dataset_info["max_height"] = max_h
    if mode == "multispectral_classification":
        metadata_path = os.path.join(dataset_path[mode], name, "multispectral_metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            dataset_info["num_channels"] = int(metadata.get("num_channels") or metadata.get("in_channels"))
        except (OSError, TypeError, ValueError, AttributeError):
            dataset_info["num_channels"] = None

    for item in dataset_items:
        image_path = item.get("image")
        if image_path:
            rel_path = os.path.relpath(image_path, user_datasets_root)
            item["image_url"] = url_for("user_dataset_files", filename=rel_path)
            if mode == "multispectral_classification":
                item["preview_url"] = url_for("multispectral_preview", filename=rel_path)
            item["display_name"] = os.path.basename(image_path)

    pager_size_options = [15, 30, 50, 80, 120]

    # Don't overload the page
    visible_items = dataset_items[pager_size*(page-1):pager_size*page]
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

    seg_datasets = _merged_datasets(user_slug, "segmentation", "Seg")
    od_datasets  = _merged_datasets(user_slug, "detection", "OD")
    cls_datasets = _merged_datasets(user_slug, "classification", "Cls")
    ms_cls_datasets = _merged_datasets(user_slug, "multispectral_classification", "MsCls")

    datasets = {
        "segmentation"  : seg_datasets,
        "detection"     : od_datasets,
        "classification": cls_datasets,
        "multispectral_classification": ms_cls_datasets
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


@app.route('/dataset_submit', methods=["POST"])
@login_required
def dataset_submit():
    mode = request.form.get('mode')

    num_classes = None
    if mode in ("classification", "multispectral_classification"):
        try:
            num_classes = int(request.form.get('num_classes'))
        except (TypeError, ValueError):
            return redirect(url_for("collections", mode=mode, msg="Invalid number of classes", msg_type="danger"))
        
    filename = secure_filename(request.form.get('dataset_zip', ''))

    user_slug = get_current_user_slug()
    zip_path = os.path.join(user_root(user_slug), "tmp_datasets_zips", filename)
    zip_path = os.path.abspath(zip_path)
    expected_dir = os.path.abspath(os.path.join(user_root(user_slug), "tmp_datasets_zips"))
    if not zip_path.startswith(expected_dir + os.sep):
        return "Forbidden", 403

    if not os.path.isfile(zip_path):
        error_msg = "Uploaded dataset zip not found"
        flash(error_msg, "danger")
        return redirect(url_for("collections", mode=mode, msg=error_msg, msg_type="danger"))

    success, msg, final_path = process_dataset(
        mode, zip_path, num_classes, user_slug=user_slug,
        num_channels=request.form.get("num_channels"),
    )

    if not success:
        return redirect(url_for("collections", mode=mode, msg=msg, msg_type="danger"))

    # HESTIA: persist the validated dataset to the data lake (non-fatal).
    if HESTIA_ENABLED and final_path:
        try:
            _persist_dataset_to_hestia(user_slug, mode, final_path, num_classes, request.form)
        except Exception as e:
            app.logger.warning(f"HESTIA dataset persist failed: {e}")

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
        "classification": cls_models,
        "multispectral_classification": cls_models
    }

    # Collections
    user_slug = get_current_user_slug()
    user_datasets_root = os.path.join(user_root(user_slug), "Datasets")

    od_collections  = _merged_datasets(user_slug, "detection", "OD")
    seg_collections = _merged_datasets(user_slug, "segmentation", "Seg")
    cls_collections = _merged_datasets(user_slug, "classification", "Cls")
    ms_cls_collections = _merged_datasets(user_slug, "multispectral_classification", "MsCls")

    collections = {
        "segmentation"  : seg_collections,
        "detection"     : od_collections,
        "classification": cls_collections,
        "multispectral_classification": ms_cls_collections
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
            "description": "Controls how fast the AI tries to learn from its mistakes. Think of it like adjusting your aim in darts: a big learning rate means making huge corrections, while a small one means making tiny, careful adjustments."
        },
        "batch_size": {
            "range"    : "discrete",
            "values"   : [1, 2, 4, 8, 16, 32, 64],
            "lower"    : 1,
            "upper"    : 64,
            "lower_def": config.get('defaults').get('bs_lower'),
            "upper_def": config.get('defaults').get('bs_upper'),
            "description": "The number of examples the model processes at the same time before updating its knowledge."
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
            "description": "How many times the AI reads through the entire set of examples. One epoch means the AI has seen every single training image exactly once."
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
            "scale" : {
                "type": "bool",
                "default": config.get('defaults').get('cls_scale'),
                "description": "Randomly scales images to different sizes for improved robustness."
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
            "flip": {
                "type"   : "bool",
                "default": config.get('defaults').get('det_flip'),
                "description": "Flips images horizontally or vertically for spatial invariance."
            },
            "rotate": {
                "type"   : "bool",
                "default": config.get('defaults').get('det_rotate'),
                "description": "Randomly rotates images to improve generalization to different orientations."
            },
            "scale" : {
                "type": "bool",
                "default": config.get('defaults').get('det_scale'),
                "description": "Randomly scales images to different sizes for improved robustness."
            },
        }
    }

    augmentations["multispectral_classification"] = augmentations["classification"]

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

    seg_csv = os.path.join(user_root(user_slug), "models_db", "trained_models_db_segm.csv")
    od_csv  = os.path.join(user_root(user_slug), "models_db", "trained_models_db_od.csv")
    cls_csv = os.path.join(user_root(user_slug), "models_db", "trained_models_db_cls.csv")
    ms_cls_csv = os.path.join(user_root(user_slug), "models_db", "trained_models_db_ms_cls.csv")
    if HESTIA_ENABLED:
        seg_models = _hestia_models_for_template(user_slug, "segmentation", seg_csv)
        od_models  = _hestia_models_for_template(user_slug, "detection", od_csv)
        cls_models = _hestia_models_for_template(user_slug, "classification", cls_csv)
        ms_cls_models = _hestia_models_for_template(user_slug, "multispectral_classification", ms_cls_csv)
    else:
        seg_models = load_models(seg_csv)
        od_models  = load_models(od_csv)
        cls_models = load_models(cls_csv)
        ms_cls_models = load_models(ms_cls_csv)

    models = {
        "segmentation"  : seg_models,
        "detection"     : od_models,
        "classification": cls_models,
        "multispectral_classification": ms_cls_models
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
    transfer_learning   = request.form.get('transfer_learning', 'true')
    selected_collection = request.form.get('collection')
    if selected_collection:
        selected_collection = secure_filename(selected_collection)

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
    in_channels = None
    if mode in ("classification", "multispectral_classification") and selected_collection:
        dataset_dir = HESTIA_DATASET_DIR[mode]
        dataset_root = os.path.join(user_root(user_slug), "Datasets", dataset_dir, selected_collection)
        train_dir = os.path.join(dataset_root, "train")
        val_dir = os.path.join(dataset_root, "val")
        cls_split_flag = os.path.isdir(train_dir) and os.path.isdir(val_dir)
        if mode == "multispectral_classification":
            if HESTIA_ENABLED:
                hc.ensure_dataset_local(user_slug, mode, selected_collection,
                                        os.path.dirname(dataset_root))
            metadata_path = os.path.join(dataset_root, "multispectral_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    in_channels = int(json.load(handle)["num_channels"])
            except (OSError, ValueError, KeyError):
                return jsonify({"error": "Invalid multispectral dataset metadata"}), 400

    cls_cmd = [
        "python", "training_button_classification.py",
        "--user", user_slug,
    ] + base_args + [
        "--blur", bool_str(blur_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
        "--scale", bool_str(scale_enabled),
        "--dataset_already_split", bool_str(cls_split_flag),
        "--transfer_learning", bool_str(transfer_learning),
    ]

    ms_cls_cmd = [
        "python", "training_button_multispectral_classification.py",
        "--user", user_slug,
    ] + base_args + [
        "--in_channels", str(in_channels),
        "--blur", bool_str(blur_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
        "--scale", bool_str(scale_enabled),
        "--dataset_already_split", bool_str(cls_split_flag),
        "--transfer_learning", bool_str(transfer_learning),
    ]

    # Detection command
    det_cmd = [
        "python", "training_button_object_detection.py",
        "--user", user_slug,
    ] + base_args + [
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
        "--scale", bool_str(scale_enabled),
        "--transfer_learning", bool_str(transfer_learning),
    ]

    subprocesses = {
        "segmentation"  : seg_cmd,
        "detection"     : det_cmd,
        "classification": cls_cmd,
        "multispectral_classification": ms_cls_cmd
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
        ],
        "multispectral_classification": [
            os.path.join(user_root(user_slug), "MultispectralClassification", "runs", "user_experiments.csv"),
            os.path.join(user_root(user_slug), "models_db", "trained_models_db_ms_cls.csv"),
            "MsCls",
            user_slug,
        ]
    }
    
    cmd = subprocesses[mode]

    job_id = str(uuid.uuid4())
    job_status = {"status": "running", "mode": mode, "last_heartbeat": time.time()}

    # HESTIA: rehydrate the dataset into the local cache (so the Katib job's
    # hostPath mount sees it) and record the experiment. Non-fatal.
    if HESTIA_ENABLED:
        try:
            dest_root = os.path.join(user_root(user_slug), "Datasets", HESTIA_DATASET_DIR[mode])
            hc.ensure_dataset_local(user_slug, mode, selected_collection, dest_root)

            ds = hc.find_dataset(user_slug, mode, selected_collection)
            dataset_id = ds.get("dataset_id") if ds else None
            if mode == "detection":
                augmentations = {"rotate": bool_str(rotate_enabled), "flip": bool_str(flip_enabled), "scale": bool_str(scale_enabled)}
            else:
                augmentations = {"blur": bool_str(blur_enabled), "scale": bool_str(scale_enabled),
                                 "rotate": bool_str(rotate_enabled), "flip": bool_str(flip_enabled)}
            params_payload = {
                "learning_rate": {"left": lr_left, "right": lr_right},
                "batch_size": {"left": bs_left, "right": bs_right},
                "epochs": {"left": epoch_left, "right": epoch_right},
                "augmentations": augmentations,
            }
            if in_channels is not None:
                params_payload["in_channels"] = in_channels
            owner_email = get_current_user_email()
            experiment_id = hc.create_experiment(
                user_slug, mode, dataset_id=dataset_id,
                dataset_name=selected_collection, requested_model=selected_model,
                params=params_payload, job_id=job_id, owner_email=owner_email,
            )
            if experiment_id:
                job_status["experiment_id"] = experiment_id
            if dataset_id:
                job_status["dataset_id"] = dataset_id
            if owner_email:
                job_status["owner_email"] = owner_email
        except Exception as e:
            app.logger.warning(f"HESTIA train setup failed: {e}")

    _write_job_status(user_root, user_slug, job_id, job_status)

    thread = threading.Thread(
        target=_run_training_job,
        args=(user_root, user_slug, job_id, cmd, mode, paths, HESTIA_ENABLED, _push_trained_model_to_hestia, app.logger),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "running", "job_id": job_id})


@app.route('/train_status/<job_id>', methods=["GET"])
@login_required
def train_status(job_id):
    user_slug = get_current_user_slug()
    status = _read_job_status(user_root, user_slug, job_id)
    if not status:
        return jsonify({"error": "job_not_found"}), 404
    status = _mark_stale_if_dead(user_root, user_slug, job_id, status)
    return jsonify(status)

@app.route('/train_jobs/active', methods=["GET"])
@login_required
def active_train_jobs():
    user_slug = get_current_user_slug()
    jobs_dir = os.path.join(user_root(user_slug), "train_jobs")
    active = []
    for path in glob.glob(os.path.join(jobs_dir, "*.json")):
        job_id = os.path.splitext(os.path.basename(path))[0]
        status = _read_job_status(user_root, user_slug, job_id)
        status = _mark_stale_if_dead(user_root, user_slug, job_id, status)
        if status and status.get("status") == "running":
            status["job_id"] = job_id
            active.append(status)
    return jsonify(active)


@app.route('/inference/results', methods=['GET'])
@login_required
def inference_results():
    raw_id = request.args.get("id")
    mode = request.args.get("mode", default="segmentation", type=str)
    if mode not in ("segmentation", "detection", "classification", "multispectral_classification"):
        return "Invalid mode", 400

    user_slug = get_current_user_slug()
    ensure_user_folders(user_slug)
    mode_params = _inference_params(user_slug)[mode]
    model, _, model_id = _resolve_inference_model(mode_params, raw_id)
    if not model:
        return "Model not found", 404

    results_runs, hestia_runs = _collect_available_inference_runs(user_slug, mode, model_id)
    color_table = (
        _load_segmentation_color_table(user_slug, model.get("trained_on"))
        if mode == "segmentation" else []
    )
    if mode == "segmentation" and hestia_runs is not None and not color_table:
        color_table = next(
            (
                (row.get("extra") or {}).get("color_table")
                for row in hestia_runs
                if (row.get("extra") or {}).get("color_table")
            ),
            [],
        )
    return render_template(
        "inference_results.html",
        mode=mode,
        model_id=model_id,
        model=model,
        metric_label=mode_params["metric"],
        color_table=color_table,
        has_inference_results=bool(results_runs),
        results_runs=results_runs,
    )


@app.route('/inference/results/vlm', methods=['POST'])
@login_required
def inference_result_vlm():
    """Create a VLM handoff for one saved classification result."""
    model_id = request.form.get("model_id", "")
    timestamp = request.form.get("timestamp", "")
    filename = request.form.get("filename", "")
    result_root = request.form.get("result_root", "classification")
    components = (model_id, timestamp, filename)
    if any(not value or secure_filename(value) != value for value in components):
        return "Invalid inference result", 400
    if result_root not in ("classification", "_hestia_results/classification"):
        return "Invalid inference result", 400

    base_name = os.path.splitext(filename)[0]
    input_path = f"{result_root}/inputs/{model_id}/{timestamp}/{filename}"
    gradcam_path = f"{result_root}/outputs/{model_id}/{timestamp}/{base_name}_gradcam.jpg"
    output_path = f"{result_root}/outputs/{model_id}/{timestamp}/{base_name}.txt"

    user_slug = get_current_user_slug()
    required_paths = (input_path, gradcam_path, output_path)
    resolved_paths = [
        _inference_handoff_path(user_slug, relative_path)
        for relative_path in required_paths
    ]
    if not all(path and os.path.isfile(path) for path in resolved_paths):
        flash("This saved result is incomplete and cannot start a VLM chat.", "warning")
        return redirect(url_for("inference_results", id=model_id, mode="classification"))

    handoff_id = _store_vlm_handoff(input_path, gradcam_path, output_path)
    return redirect(url_for("vlm_chat", handoff_id=handoff_id))


@app.route('/inference', methods=['GET', 'POST'])
@login_required
def inference():
    raw_id = request.args.get("id")

    mode = request.args.get("mode", default="segmentation", type=str)
    if mode not in ("segmentation", "detection", "classification", "multispectral_classification"):
        return "Invalid mode", 400

    user_slug = get_current_user_slug()
    ensure_user_folders(user_slug)

    params = _inference_params(user_slug)
    mode_params = params[mode]
    metric = mode_params["metric"]

    model, hestia_model, model_id = _resolve_inference_model(mode_params, raw_id)
    if not model:
        return "Model not found", 404

    error_msg   = None
    success_msg = None

    results         = []
    model_name      = model['name']
    checkpoint_path = model.get('checkpoint_path')   # set via rehydrate for HESTIA models
    config_path     = model.get('config_path')
    color_table     = []                        # for color map (segmentation mode)
    dataset_name    = model.get('trained_on')
    inference_root = os.path.join(user_root(user_slug), "inference", mode)
    
    if request.method == 'POST':
        if mode == "segmentation":
            # HESTIA: rehydrate model weights/config + dataset labelmap into the
            # local cache so the inference container (and color table) can see them.
            if hestia_model:
                cache_dir = os.path.join(user_root(user_slug), "Segmentation",
                                         "_hestia_cache", str(model_id))
                checkpoint_path, config_path = hc.ensure_model_local(hestia_model, cache_dir)
                if dataset_name:
                    hc.ensure_dataset_local(
                        user_slug, "segmentation", dataset_name,
                        os.path.join(user_root(user_slug), "Datasets", "Segmentation"))

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
                        "chown", "-R", chown_target(config), f"/data/{user_slug}/inference/segmentation/outputs/"
                    ])

                finally:
                    container.stop()
                    container.remove()

                # Get all output files
                output_data_dir = output_dir
                output_files    = glob.glob(os.path.join(output_data_dir, '*.png')) + glob.glob(os.path.join(output_data_dir, '*.jpg'))
                
                # Create a mapping of output files by name (without extension)
                output_map = {}
                for f in output_files:
                    base = os.path.splitext(os.path.basename(f))[0]
                    output_map[base] = f

                # Process each uploaded file and match with output
                for file in files:
                    if file.filename:
                        filename   = secure_filename(file.filename)
                        input_image = url_for("user_inference_files", filename=f"segmentation/inputs/{model_id}/{timestamp}/{filename}")
                        
                        # Find matching output file
                        output_image = None
                        base_name    = os.path.splitext(filename)[0]

                        if base_name in output_map:
                            output_path     = output_map[base_name]
                            output_filename = os.path.basename(output_path)
                            output_image = url_for("user_inference_files", filename=f"segmentation/outputs/{model_id}/{timestamp}/{output_filename}")
                        else:
                            print(f"WARNING: No output found for {filename}")

                        results.append({
                            'input_image' : input_image,
                            'output_image': output_image
                        })

                success_msg = f"Inference for {len(files)} image(s) completed!"

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

            # HESTIA: persist inference inputs + outputs (non-fatal).
            if HESTIA_ENABLED and hestia_model and files:
                try:
                    _persist_inference_to_hestia(
                        user_slug, "segmentation", str(model_id), model_name,
                        dataset_name, files, save_dir, output_map, color_table)
                except Exception as e:
                    app.logger.warning(f"HESTIA inference persist failed: {e}")

        if mode == "detection":
            # HESTIA: rehydrate model weights/config into the local cache so the
            # inference container can see them.
            if hestia_model:
                cache_dir = os.path.join(user_root(user_slug), "ObjectDetection",
                                         "_hestia_cache", str(model_id))
                checkpoint_path, config_path = hc.ensure_model_local(hestia_model, cache_dir)

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
                        "python", "/data/ObjectDetection/ultralytics/inference_script.py",
                        "--checkpoint", checkpoint_path,
                        "--input", f"/data/{user_slug}/inference/detection/inputs/{model_id}/{timestamp}/",
                        "--output", f"/data/{user_slug}/inference/detection/outputs/{model_id}/{timestamp}/",
                    ])

                    container.exec_run([
                        "chown", "-R", chown_target(config), "/data/"
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

                # HESTIA: persist inference inputs + outputs (non-fatal).
                if HESTIA_ENABLED and hestia_model:
                    try:
                        det_out_map = {os.path.basename(v): v for v in output_map.values()}
                        _persist_inference_to_hestia(
                            user_slug, "detection", str(model_id), model_name,
                            dataset_name, files, save_dir, det_out_map, None)
                    except Exception as e:
                        app.logger.warning(f"HESTIA inference persist failed: {e}")

        if mode == "classification":
            # HESTIA: rehydrate model weights/config into the local cache so the
            # inference container can see them.
            if hestia_model:
                cache_dir = os.path.join(user_root(user_slug), "Classification",
                                         "_hestia_cache", str(model_id))
                checkpoint_path, config_path = hc.ensure_model_local(hestia_model, cache_dir)

            files = [f for f in request.files.getlist('file') if f.filename]  # Get multiple files
            if files:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                save_dir = os.path.join(inference_root, "inputs", str(model_id), timestamp)
                output_dir = os.path.join(inference_root, "outputs", str(model_id), timestamp)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                # Save all uploaded files first
                filenames = []
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(save_dir, filename)
                        file.save(filepath)
                        filenames.append(filename)

                # Single container for the whole batch
                container = client.containers.run(
                    IMAGE_SEGM_CLS,
                    command="sleep infinity",
                    detach=True,
                    volumes={BASE_HOST_PATH_OUT: {"bind": "/data", "mode": "rw"}}
                )

                try:
                    for filename in filenames:
                        # Run inference
                        container.exec_run([
                            "python", "/data/Classification/inference.py",
                            "--model_path", checkpoint_path,
                            "--model", model_name,
                            "--class_names", config_path,
                            "--image", f"/data/{user_slug}/inference/classification/inputs/{model_id}/{timestamp}/{filename}",
                            "--output_dir", f"/data/{user_slug}/inference/classification/outputs/{model_id}/{timestamp}/"
                        ])

                    container.exec_run([
                        "chown", "-R", chown_target(config), f"/data/{user_slug}/inference/classification/outputs/"
                    ])

                finally:
                    container.stop()
                    container.remove()

                # Collect results for each file after the batch has run
                for filename in filenames:
                    input_file = url_for("user_inference_files", filename=f"classification/inputs/{model_id}/{timestamp}/{filename}")

                    # Find the corresponding output file by matching the base name
                    output_data_dir  = output_dir
                    base_name        = os.path.splitext(filename)[0]  # Get filename without extension
                    expected_output  = f"{base_name}.txt"
                    output_file_path = os.path.join(output_data_dir, expected_output)
                    
                    gradcam_filename = f"{base_name}_gradcam.jpg"
                    gradcam_path = os.path.join(output_data_dir, gradcam_filename)

                    gradcam_file = None
                    if os.path.exists(gradcam_path):
                        gradcam_file = url_for(
                            "user_inference_files",
                            filename=f"classification/outputs/{model_id}/{timestamp}/{gradcam_filename}"
                        )

                    output_text = None
                    if os.path.exists(output_file_path):
                        # Read the output text
                        with open(output_file_path, 'r', encoding='utf-8') as f:
                            output_text = f.read()
                    else:
                        print(f"WARNING: Output file not found for {filename}: {output_file_path}")
                    
                    handoff_id = _store_vlm_handoff(
                        f"classification/inputs/{model_id}/{timestamp}/{filename}",
                        (
                            f"classification/outputs/{model_id}/{timestamp}/{gradcam_filename}"
                            if gradcam_file else None
                        ),
                        f"classification/outputs/{model_id}/{timestamp}/{expected_output}",
                    )

                    results.append({
                        'input_file' : input_file,
                        'output_text': output_text,
                        'gradcam_file': gradcam_file,
                        'filename'   : filename,
                        'vlm_handoff_id': handoff_id,
                    })

                # HESTIA: persist inference inputs + outputs (non-fatal).
                if HESTIA_ENABLED and hestia_model:
                    try:
                        cls_output_paths = (
                            glob.glob(os.path.join(output_dir, '*.txt'))
                            + glob.glob(os.path.join(output_dir, '*_gradcam.jpg'))
                        )
                        cls_out_map = {os.path.basename(p): p for p in cls_output_paths}
                        _persist_inference_to_hestia(
                            user_slug, "classification", str(model_id), model_name,
                            dataset_name, files, save_dir, cls_out_map, None)
                    except Exception as e:
                        app.logger.warning(f"HESTIA inference persist failed: {e}")

                success_msg = f"Inference for {len(files)} image(s) completed!"

        if mode == "multispectral_classification":
            if hestia_model:
                cache_dir = os.path.join(user_root(user_slug), "MultispectralClassification",
                                         "_hestia_cache", str(model_id))
                checkpoint_path, config_path = hc.ensure_model_local(hestia_model, cache_dir)

            files = [
                uploaded for uploaded in request.files.getlist("file")
                if uploaded.filename and os.path.splitext(uploaded.filename)[1].lower() in (".tif", ".tiff")
            ]
            if not files:
                error_msg = "Upload one or more multiband TIFF files."
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                save_dir = os.path.join(inference_root, "inputs", str(model_id), timestamp)
                output_dir = os.path.join(inference_root, "outputs", str(model_id), timestamp)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                filenames = []
                for uploaded in files:
                    filename = secure_filename(uploaded.filename)
                    uploaded.save(os.path.join(save_dir, filename))
                    filenames.append(filename)

                container = client.containers.run(
                    IMAGE_SEGM_CLS,
                    command="sleep infinity",
                    detach=True,
                    volumes={BASE_HOST_PATH_OUT: {"bind": "/data", "mode": "rw"}},
                )
                try:
                    for filename in filenames:
                        execution = container.exec_run([
                            "python", "/data/MultispectralClassification/inference.py",
                            "--model_path", checkpoint_path,
                            "--config", config_path,
                            "--image", f"/data/{user_slug}/inference/multispectral_classification/inputs/{model_id}/{timestamp}/{filename}",
                            "--output_dir", f"/data/{user_slug}/inference/multispectral_classification/outputs/{model_id}/{timestamp}/",
                        ])
                        if execution.exit_code != 0:
                            error_msg = "One or more TIFF files do not match this model's spectral input."
                    container.exec_run([
                        "chown", "-R", chown_target(config),
                        f"/data/{user_slug}/inference/multispectral_classification/outputs/",
                    ])
                finally:
                    container.stop()
                    container.remove()

                for filename in filenames:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
                    if not os.path.isfile(output_path):
                        continue
                    with open(output_path, "r", encoding="utf-8") as handle:
                        output_text = handle.read()
                    results.append({
                        "input_file": url_for(
                            "user_inference_files",
                            filename=f"multispectral_classification/inputs/{model_id}/{timestamp}/{filename}",
                        ),
                        "preview_file": url_for(
                            "multispectral_inference_preview",
                            filename=f"multispectral_classification/inputs/{model_id}/{timestamp}/{filename}",
                        ),
                        "output_text": output_text,
                        "filename": filename,
                    })

                if HESTIA_ENABLED and hestia_model and results:
                    try:
                        output_map = {
                            os.path.basename(path): path
                            for path in glob.glob(os.path.join(output_dir, "*.txt"))
                        }
                        _persist_inference_to_hestia(
                            user_slug, "multispectral_classification", str(model_id),
                            model_name, dataset_name, files, save_dir, output_map, None,
                        )
                    except Exception as exc:
                        app.logger.warning("HESTIA inference persist failed: %s", exc)

                if results:
                    success_msg = f"Inference for {len(results)} TIFF file(s) completed!"
    
    if success_msg: 
        flash(success_msg, "info")
    if error_msg:
        flash(error_msg, "danger")

    # Metadata-only check; artifacts are fetched lazily on the results page.
    has_inference_results = _has_available_inference_results(
        user_slug, mode, model_id
    )

    return render_template(
        'inference.html',
        mode         = mode,
        model_id     = model_id,
        model        = model,
        results      = results,
        metric_label = metric,
        color_table  = color_table,
        has_inference_results = has_inference_results,
    )

 
@app.route('/vlm-chat', methods=['GET', 'POST'])
@login_required
def vlm_chat():
    answer = None
    images_data = []
    filled_system_prompt = None
    filled_user_prompt = None
    user_slug = get_current_user_slug()
    handoff_id = request.values.get("handoff_id", "")
    handoff = session.get("vlm_handoffs", {}).get(handoff_id)
    prefill = {key: "" for key, _ in USER_PROMPT_FIELDS}
    class_probabilities = []
    handoff_images = []

    if handoff:
        output_path = _inference_handoff_path(user_slug, handoff.get("output_path", ""))
        if output_path and os.path.isfile(output_path):
            with open(output_path, "r", encoding="utf-8") as output_file:
                classification = _parse_classification_output(output_file.read())
            prefill["predicted_class"] = classification["predicted_class"]
            prefill["confidence"] = classification["confidence"]
            class_probabilities = classification["probabilities"]

        for key in ("input_path", "gradcam_path"):
            relative_path = handoff.get(key)
            image_path = _inference_handoff_path(user_slug, relative_path) if relative_path else None
            if image_path and os.path.isfile(image_path):
                handoff_images.append(url_for("user_inference_files", filename=relative_path))
            else:
                handoff_images.append(None)

        if not all(handoff_images):
            flash("The saved image or Grad-CAM is no longer available. Please upload both images manually.", "warning")
            handoff = None
            handoff_images = []

    blanks_values = {
        key: request.form.get(key, "") if request.method == "POST" else ""
        for key, _ in SYSTEM_PROMPT_BLANKS
    }

    if request.method == 'POST':
        image_sources = []
        uploaded_images = [request.files.get('image1'), request.files.get('image2')]
        handoff_paths = [
            handoff.get("input_path") if handoff else None,
            handoff.get("gradcam_path") if handoff else None,
        ]

        for uploaded_image, relative_path in zip(uploaded_images, handoff_paths):
            if uploaded_image and uploaded_image.filename:
                mime_type = uploaded_image.mimetype or "image/jpeg"
                image_sources.append((mime_type, uploaded_image.read()))
                continue

            image_path = _inference_handoff_path(user_slug, relative_path) if relative_path else None
            if image_path and os.path.isfile(image_path):
                mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
                with open(image_path, "rb") as image_file:
                    image_sources.append((mime_type, image_file.read()))

        if len(image_sources) != 2:
            flash("Please provide both the original image and its Grad-CAM visualization.", "error")
        else:
            content = []
            for mime_type, image_bytes in image_sources:
                img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                images_data.append({"mime_type": mime_type, "image_b64": img_b64})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}
                })
 
            filled_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**blanks_values)
            filled_user_prompt = build_user_prompt(request.form)
            content.append({"type": "text", "text": filled_user_prompt})
 
            payload = {
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": filled_system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 512
            }
 
            try:
                resp = requests.post(VLM_URL, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                    answer = data["choices"][0]["message"]["content"]
                else:
                    flash(f"Unexpected VLM response: {data}", "error")
            except requests.exceptions.ConnectionError:
                flash("The analysis service is currently offline. Please contact an administrator to start it.", "error")
            except requests.exceptions.RequestException as e:
                flash(f"VLM request failed: {e}", "error")
            except ValueError:
                flash("VLM returned invalid JSON", "error")
 
    return render_template(
        'vlm_chat.html',
        answer=answer,
        images_data=images_data,
        filled_system_prompt=filled_system_prompt,
        filled_user_prompt=filled_user_prompt,
        blanks=SYSTEM_PROMPT_BLANKS,
        user_fields=USER_PROMPT_FIELDS,
        class_examples=USER_PROMPT_CLASS_EXAMPLES,
        handoff_id=handoff_id if handoff else "",
        handoff_images=handoff_images,
        prefill=prefill,
        class_probabilities=class_probabilities,
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8056)