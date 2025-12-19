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
from utils.models_page import write_results
from utils.file_management import remove_unnecessary_files
from utils.helpers import load_datasets, load_models_available, load_dataset_info, get_max_image_size, load_models
from utils.load_config import load_config
import docker
import zipfile
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


client = docker.from_env()

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config.from_file('config.toml', load=tomllib.load, text=False)

# Login requirements
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Dummy user
class User(UserMixin):
    id = 1
    username = "user"
    password = "1234"  
    full_name = "Guest User"

user_instance = User()

@login_manager.user_loader
def load_user(user_id):
    if str(user_id) == "1":
        return user_instance
    return None

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == user_instance.username and password == user_instance.password:
            login_user(user_instance)
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
BASE_DATASET_PATH   = config.get("paths").get("base_dataset_path")
BASE_INFERENCE_PATH = config.get("paths").get("base_inference_path")
IMAGE_SEGM_CLS  = config.get("images").get("classification")
IMAGE_OD        = config.get("images").get("detection")


@app.route(f'/{BASE_DATASET_PATH}/<path:filename>')
def data_files(filename):
    """Serve files from the repository `data/` directory.

    Use `url_for('data_files', filename='path/inside/data.jpg')` from templates.
    This includes a path-check to prevent directory traversal.
    """
    data_dir = os.path.abspath(os.path.join(app.root_path, f'{BASE_DATASET_PATH}'))
    requested = os.path.abspath(os.path.join(data_dir, filename))

    # Prevent path traversal: requested must be inside data_dir
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
    
    dataset_path = {
        "segmentation"  : f"{BASE_DATASET_PATH}/Segmentation/",
        "detection"     : f"{BASE_DATASET_PATH}/Object-Detection/",
        "classification": f"{BASE_DATASET_PATH}/Classification/"
    }

    dataset_info, dataset_items = load_dataset_info(
        filepath = dataset_path[mode],
        name     = name,
        mode     = mode
    )

    max_w, max_h = get_max_image_size(dataset_items)
    dataset_info["max_width"]  = max_w
    dataset_info["max_height"] = max_h

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
        dataset_path_seg   = f"{BASE_DATASET_PATH}/Segmentation/",
        dataset_path_det   = f"{BASE_DATASET_PATH}/Object-Detection/",
        dataset_path_cls   = f"{BASE_DATASET_PATH}/Classification/",
    )



# Datasets page showing all available datasets with metadata
@app.route('/collections')
@login_required
def collections():
    # Dataset logic
    seg_datasets = load_datasets(f"{BASE_DATASET_PATH}/Segmentation", "Seg")
    od_datasets  = load_datasets(f"{BASE_DATASET_PATH}/Object-Detection", "OD")
    cls_datasets = load_datasets(f"{BASE_DATASET_PATH}/Classification", "Cls")

    def enrich(datasets, base_path):
        enriched = []
        for d in datasets:
            folder_path = os.path.join(base_path, d["id"])
            creation_timestamp = os.path.getctime(folder_path)
            creation_date = datetime.fromtimestamp(creation_timestamp).strftime("%d/%m/%Y") # fixes the format to be the same with inference page format
            enriched.append({
                "name": d["name"],
                "num_samples": d["num_samples"],
                "type": "2D images",
                "date": creation_date,
                "url": "/dataset"
            })
        return enriched

    seg_datasets = enrich(seg_datasets, f"{BASE_DATASET_PATH}/Segmentation")
    od_datasets  = enrich(od_datasets, f"{BASE_DATASET_PATH}/Object-Detection")
    cls_datasets = enrich(cls_datasets, f"{BASE_DATASET_PATH}/Classification")

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
    save_dir = os.path.join('data', 'media', 'datasets_zips')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    try:
        file.save(path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save file: {e}'}), 500
    return jsonify({'status': 'ok', 'filename': filename})


def process_dataset(mode, zip_path, num_classes=None):
    DEST_PATHS = {
        "segmentation"  : f"{BASE_DATASET_PATH}/Segmentation",
        "detection"     : f"{BASE_DATASET_PATH}/Object-Detection",
        "classification": f"{BASE_DATASET_PATH}/Classification"
    }

    # unzip to a temp folder
    tmp_root = "data/tmp_datasets"
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

    zip_path = os.path.join("data", "media", "datasets_zips", filename)

    if not os.path.isfile(zip_path):
        error_msg = "Uploaded dataset zip not found"
        flash(error_msg, "danger")
        return redirect(url_for("collections", mode=mode, msg=error_msg, msg_type="danger"))

    success, msg = process_dataset(mode, zip_path, num_classes)

    if not success:
        return redirect(url_for("collections", mode=mode, msg=msg, msg_type="danger"))

    return redirect(url_for("collections", mode=mode, msg=msg, msg_type="info"))

@app.route('/train_model', methods=['GET'])
@login_required
def train_model():
    # Models
    seg_models = load_models_available('data/models_available_segmentation.csv')
    cls_models = load_models_available('data/models_available_classification.csv')
    od_models  = load_models_available('data/models_available_object_detection.csv')

    models = {
        "segmentation"  : seg_models,
        "detection"     : od_models,
        "classification": cls_models
    }

    # Collections
    od_collections  = load_datasets(f"{BASE_DATASET_PATH}/Object-Detection","OD")
    seg_collections = load_datasets(f"{BASE_DATASET_PATH}/Segmentation","Seg")
    cls_collections = load_datasets(f"{BASE_DATASET_PATH}/Classification","Cls")

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
            "upper_def": config.get('defaults').get('lr_upper')
        },
        "batch_size": {
            "range"    : "discrete",
            "values"   : [1, 2, 4, 8, 16, 32, 64],
            "lower"    : 1,
            "upper"    : 64,
            "lower_def": config.get('defaults').get('bs_lower'),
            "upper_def": config.get('defaults').get('bs_upper')
        },
        "epochs": {
            "range"    : "continuous",
            "min"      : 1,
            "max"      : 100,
            "step"     : 1,
            "lower"    : 1,
            "upper"    : 100,
            "lower_def": config.get('defaults').get('ep_lower'),
            "upper_def": config.get('defaults').get('ep_upper')
        },
    }

    augmentations = {
        "segmentation": {
            "blur"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('seg_blur')
            },
            "scale" : {
                "type": "bool",
                "default": config.get('defaults').get('seg_scale')
            },
            "rotate": {
                "type": "bool",
                "default": config.get('defaults').get('seg_rotate')
            },
            "flip"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('seg_flip')
            }
        },
        "classification": {
            "blur"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_blur')
            },
            "rotate": {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_rotate')
            },
            "flip"  : {
                "type"   : "bool",
                "default": config.get('defaults').get('cls_flip')
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
                "unit"           : None
            },
            "saturation": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "max_val_default": config.get('defaults').get('det_sat'),
                "unit"           : None
            },
            "value": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "max_val_default": config.get('defaults').get('det_val'),
                "unit"           : None
            },
            "flip": {
                "type"           : "range",
                "default"        : True,
                "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                "prob_default"   : config.get('defaults').get('det_flip'),
                "unit"           : None
            },
            "rotate": {
                "type"           : "range",
                "default"        : True,
                # "prob_range"     : [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                # "prob_default"   : 0.2,
                "max_val_range"  : [0, 30, 60, 90, 120, 150, 180],
                "max_val_default": config.get('defaults').get('det_rotate'),
                "unit"           : "degrees"
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
    seg_models = load_models('data/trained_models_db_segm.csv')
    od_models  = load_models('data/trained_models_db_od.csv')
    cls_models = load_models('data/trained_models_db_cls.csv')

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
    ] + base_args + [
        "--blur", bool_str(blur_enabled),
        "--scale", bool_str(scale_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
    ]

    # Classification command
    cls_cmd = [
        "python", "training_button_classification.py",
    ] + base_args + [
        "--blur", bool_str(blur_enabled),
        "--rotate", bool_str(rotate_enabled),
        "--flip", bool_str(flip_enabled),
    ]

    # Detection command
    det_cmd = [
        "python", "training_button_object_detection.py",
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
            f"{BASE_HOST_PATH}/Segmentation/runs/user_experiments.csv",
            "data/trained_models_db_segm.csv",
            "Seg"
        ],
        "detection" : [
            f"{BASE_HOST_PATH}/ObjectDetection/runs/user_experiments.csv",
            "data/trained_models_db_od.csv",
            "OD"
        ],
        "classification": [
            f"{BASE_HOST_PATH}/Classification/runs/user_experiments.csv",
            'data/trained_models_db_cls.csv',
            "Cls"
        ]
    }
    
    process = subprocess.run(subprocesses[mode])

    success_msg = None
    error_msg   = None

    if process.returncode != 0:
        error_msg = f"Training failed: {process.returncode}"
        flash(error_msg, "danger")
    else:
        success_msg = f"Training successful!"
        write_results(*paths[mode])
        flash(success_msg, "info")
    
    # flash is omega bugged on redirect, we flash
    return redirect(url_for(
        "models",
        mode=mode,
        msg=success_msg if success_msg is not None else error_msg,
        msg_type="danger" if error_msg is not None else "info"
    ))


@app.route('/inference', methods=['GET', 'POST'])
@login_required
def inference():
    model_id = int(request.args.get("id"))
    mode     = request.args.get("mode")

    # Init params
    params = {
        "segmentation": {
            "csv"   : 'data/trained_models_db_segm.csv',
            "metric": "mIoU Score"
        },
        "detection": {
            "csv"   : 'data/trained_models_db_od.csv',
            "metric": "mAP 50-95 Score"
        },
        "classification": {
            "csv"   : 'data/trained_models_db_cls.csv',
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
    
    if request.method == 'POST':
        if mode == "segmentation":
            files = request.files.getlist('image')  # Get multiple files
            if files:
                save_dir = f"{BASE_HOST_PATH}/Segmentation/for_inference/{model_id}"
                os.makedirs(save_dir, exist_ok=True)
                
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
                        "-i", f"/data/Segmentation/for_inference/{model_id}/",
                        "-o", f"/data/Segmentation/inference_outputs/{model_id}/"
                    ])

                    container.exec_run([
                        "chown", "-R", "1000:1000", "/data/Segmentation/inference_outputs/"
                    ])

                finally:
                    container.stop()
                    container.remove()

                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                static_input_dir  = f"static/media/for_inference/segmentation/{model_id}/{timestamp}"
                static_output_dir = f"static/media/outputs/segmentation/{model_id}/{timestamp}"
                os.makedirs(static_input_dir, exist_ok=True)
                os.makedirs(static_output_dir, exist_ok=True)

                # Get all output files
                output_data_dir = f"{BASE_HOST_PATH}/Segmentation/inference_outputs/{model_id}/"
                output_files    = glob.glob(os.path.join(output_data_dir, '*.png')) + glob.glob(os.path.join(output_data_dir, '*.jpg'))
                
                # Create a mapping of output files by name
                output_map = {os.path.basename(f): f for f in output_files}
                
                # Process each uploaded file and match with output
                for file in files:
                    if file.filename:
                        filename   = secure_filename(file.filename)
                        input_path = os.path.join(save_dir, filename)
                        
                        # Copy input to static
                        shutil.copy(input_path, os.path.join(static_input_dir, filename))
                        input_image = url_for('static', filename=f'media/for_inference/segmentation/{model_id}/{timestamp}/{filename}')
                        
                        # Find matching output file
                        output_image = None
                        base_name    = os.path.splitext(filename)[0]
                        for output_name, output_path in output_map.items():
                            if base_name in output_name:
                                shutil.copy(output_path, os.path.join(static_output_dir, output_name))
                                output_image = url_for('static', filename=f'media/outputs/segmentation/{model_id}/{timestamp}/{output_name}')
                                break
                        
                        results.append({
                            'input_image' : input_image,
                            'output_image': output_image
                        })
            
            # Load datasets coloring to propagate it to the inference page
            if dataset_name:
                dataset_file = f"{BASE_DATASET_PATH}/Segmentation/{dataset_name}/labelmap.txt"

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
                remove_unnecessary_files(f"{BASE_HOST_PATH}/Segmentation/for_inference") 
                remove_unnecessary_files(f"{BASE_HOST_PATH}/Segmentation/inference_outputs")
        
        if mode == "detection":
            files = request.files.getlist('image')  # Get multiple files
            if files:
                save_dir = f"{BASE_HOST_PATH}/ObjectDetection/for_inference_od/od{model_id}" 
                os.makedirs(save_dir, exist_ok=True)
                
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
                        "--input", f"/data/ObjectDetection/for_inference_od/od{model_id}/",
                        "--output", f"/data/ObjectDetection/outputs/{model_id}/",
                    ])

                    # ownership fix
                    container.exec_run([
                        "chown", "-R", "1000:1000", "/data/"
                    ])

                finally:
                    container.stop()
                    container.remove()


                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                static_input_dir  = f"static/media/for_inference/detection/{model_id}/{timestamp}"
                static_output_dir = f"static/media/outputs/detection/{model_id}/{timestamp}"
                os.makedirs(static_input_dir, exist_ok=True)
                os.makedirs(static_output_dir, exist_ok=True)

                # Collect outputs from predict directory
                output_data_dir = f"{BASE_HOST_PATH}/ObjectDetection/outputs/{model_id}/predict/"
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
                        input_path = os.path.join(save_dir, filename)
                        
                        # Copy input to static
                        shutil.copy(input_path, os.path.join(static_input_dir, filename))
                        input_image = url_for('static', filename=f'media/for_inference/detection/{model_id}/{timestamp}/{filename}')
                        
                        # Find matching output file
                        output_image = None
                        base_name    = os.path.splitext(filename)[0]
                        
                        if base_name in output_map:
                            output_path     = output_map[base_name]
                            output_filename = os.path.basename(output_path)
                            shutil.copy(output_path, os.path.join(static_output_dir, output_filename))
                            output_image = url_for('static', filename=f'media/outputs/detection/{model_id}/{timestamp}/{output_filename}')
                        else:
                            print(f"WARNING: No output found for {filename}")
                        
                        results.append({
                            'input_image' : input_image,
                            'output_image': output_image
                        })
                
                success_msg = f"Inference for {len(files)} image(s) completed!"
                remove_unnecessary_files(f"{BASE_HOST_PATH}/ObjectDetection/for_inference_od")
                remove_unnecessary_files(f"{BASE_HOST_PATH}/ObjectDetection/outputs")

        if mode == "classification":
            files = request.files.getlist('file')  # Get multiple files
            if files:
                save_dir = f"{BASE_HOST_PATH}/Classification/for_inference/{model_id}"
                os.makedirs(save_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                static_input_dir  = f"static/media/for_inference/classification/{model_id}/{timestamp}"
                static_output_dir = f"static/media/outputs/classification/{model_id}/{timestamp}"
                os.makedirs(static_input_dir, exist_ok=True)
                os.makedirs(static_output_dir, exist_ok=True)

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
                            "--image", f"/data/Classification/for_inference/{model_id}/{filename}",
                            "--output_dir", f"/data/Classification/inference_outputs/{model_id}/"
                        ])

                        # Fix permissions
                        container.exec_run([
                            "chown", "-R", "1000:1000", "/data/Classification/inference_outputs/"
                        ])

                    finally:
                        container.stop()
                        container.remove()

                    # Copy input to static
                    shutil.copy(filepath, os.path.join(static_input_dir, filename))
                    input_file = url_for('static', filename=f'media/for_inference/classification/{model_id}/{timestamp}/{filename}')

                    # Find the corresponding output file by matching the base name
                    output_data_dir  = f"{BASE_HOST_PATH}/Classification/inference_outputs/{model_id}/"
                    base_name        = os.path.splitext(filename)[0]  # Get filename without extension
                    expected_output  = f"{base_name}.txt"
                    output_file_path = os.path.join(output_data_dir, expected_output)
                    
                    output_text = None
                    if os.path.exists(output_file_path):
                        # Copy output to static
                        shutil.copy(output_file_path, static_output_dir)
                        
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
            remove_unnecessary_files(f"{BASE_HOST_PATH}/Classification/for_inference") 
            remove_unnecessary_files(f"{BASE_HOST_PATH}/Classification/inference_outputs")
    
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
    app.run(debug=True, host='0.0.0.0', port=5000)