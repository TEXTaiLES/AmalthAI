import os
from flask_login import current_user

BASE_HOST_PATH = None


def init(base_host_path):
    global BASE_HOST_PATH
    BASE_HOST_PATH = base_host_path


def safe_user_slug(email):
    base = email.split("@", 1)[0].strip().lower()
    return "".join(c for c in base if c.isalnum() or c in ("-", "_"))


def user_root(slug):
    return os.path.join(BASE_HOST_PATH, slug)


def get_current_user_slug():
    slug = getattr(current_user, "slug", None)
    return slug or "guest"


def get_current_user_email():
    return getattr(current_user, "email", None)


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