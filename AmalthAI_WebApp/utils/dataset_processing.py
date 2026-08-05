import os
import shutil
from datetime import datetime

from utils.helpers import load_datasets, get_best_timestamp
from utils.user_paths import user_root, get_current_user_slug
from utils.zip_utils import safe_extract_zip
from utils import hestia_client as hc

HESTIA_ENABLED = None
HESTIA_DATASET_DIR = None


def init(hestia_enabled, hestia_dataset_dir):
    global HESTIA_ENABLED, HESTIA_DATASET_DIR
    HESTIA_ENABLED = hestia_enabled
    HESTIA_DATASET_DIR = hestia_dataset_dir


def _merged_datasets(user_slug, mode, mode_short):
    """Datasets for the UI = local cache ∪ HESTIA (authoritative), keyed by name.

    Local entries keep their on-disk sample count + folder date. HESTIA-only
    datasets (not cached on this node) are still listed and selectable; training
    or viewing rehydrates them on demand. Falls back to local-only if HESTIA is
    unreachable, so nothing (incl. legacy local-only datasets) ever disappears.
    """
    base = os.path.join(user_root(user_slug), "Datasets", HESTIA_DATASET_DIR[mode])
    out = {}
    for d in load_datasets(base, mode_short):
        folder = os.path.join(base, d["id"])
        try:
            date = datetime.fromtimestamp(get_best_timestamp(folder)).strftime("%d/%m/%Y")
        except Exception:
            date = ""
        out[d["name"]] = {"id": d["name"], "name": d["name"],
                          "num_samples": d["num_samples"], "type": "2D images",
                          "date": date, "cached": True}

    if HESTIA_ENABLED:
        rows = hc.list_datasets(user_slug, mode)
        if rows is not None:  # None => HESTIA unreachable; keep local-only
            for r in rows:
                name = r.get("name")
                if not name or name in out:
                    continue
                counts = (r.get("manifest") or {}).get("counts") or {}
                num = sum(v for v in counts.values() if isinstance(v, int)) or "—"
                date = ""
                if r.get("created_at"):
                    try:
                        date = datetime.fromisoformat(r["created_at"]).strftime("%d/%m/%Y")
                    except Exception:
                        date = ""
                out[name] = {"id": name, "name": name, "num_samples": num,
                             "type": "2D images", "date": date, "cached": False}

    return sorted(out.values(), key=lambda d: str(d["name"]).lower())


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
        safe_extract_zip(zip_path, tmp_dir)
    except Exception as e:
        return False, f"Failed to unzip: {e}", None

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
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, f"Missing: {r}", None
        
        # Check that images/train and masks/train match
        img_train = os.path.join(tmp_dir, "images/train")
        msk_train = os.path.join(tmp_dir, "masks/train")

        ok, img_set, msk_set = same_files_no_ext(img_train, msk_train)
        if not ok:
            missing_masks = img_set - msk_set
            missing_images = msk_set - img_set
            return False, f"Train mismatch. Images missing masks: {missing_masks}, masks missing images: {missing_images}", None

        # Check that images/val and masks/val match
        img_val = os.path.join(tmp_dir, "images/val")
        msk_val = os.path.join(tmp_dir, "masks/val")

        ok, img_set, msk_set = same_files_no_ext(img_val, msk_val)
        if not ok:
            missing_masks = img_set - msk_set
            missing_images = msk_set - img_set
            return False, f"Val mismatch. Images missing masks: {missing_masks}, masks missing images: {missing_images}", None

    elif mode == "detection":
        if not os.path.isfile(os.path.join(tmp_dir, "data.yaml")):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, "Missing data.yaml", None
        
        # Check OD train split
        train_img = os.path.join(tmp_dir, "train/images")
        train_lbl = os.path.join(tmp_dir, "train/labels")

        if not os.path.exists(train_img) or not os.path.exists(train_lbl):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, "Invalid YOLO structure: missing train/images or train/labels", None

        ok, img_set, lbl_set = same_files_no_ext(train_img, train_lbl)
        if not ok:
            missing_labels = img_set - lbl_set
            missing_images = lbl_set - img_set
            return False, f"Train mismatch. Images missing labels: {missing_labels}, labels missing images: {missing_images}", None

        # Check OD val split
        val_img = os.path.join(tmp_dir, "valid/images")
        val_lbl = os.path.join(tmp_dir, "valid/labels")

        if not os.path.exists(val_img) or not os.path.exists(val_lbl):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, "Invalid YOLO structure: missing valid/images or valid/labels", None

        ok, img_set, lbl_set = same_files_no_ext(val_img, val_lbl)
        if not ok:
            missing_labels = img_set - lbl_set
            missing_images = lbl_set - img_set
            return False, f"Val mismatch. Images missing labels: {missing_labels}, labels missing images: {missing_images}", None

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
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, train_classes, None

            ok, val_classes = validate_class_dir(val_dir, "val")
            if not ok:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, val_classes, None

            if train_classes != val_classes:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                missing_train = val_classes - train_classes
                missing_val = train_classes - val_classes
                return False, (
                    "Train/val class mismatch. "
                    f"Missing in train: {missing_train}, missing in val: {missing_val}"
                ), None
        else:
            subdirs = [
                d for d in os.listdir(tmp_dir)
                if os.path.isdir(os.path.join(tmp_dir, d))
            ]
            if len(subdirs) != int(num_classes):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, f"Expected {num_classes} class folders, found {len(subdirs)}", None
    
            for cls in subdirs:
                cls_path = os.path.join(tmp_dir, cls)
                contents = os.listdir(cls_path)

                # Must NOT be empty
                if len(contents) == 0:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return False, f"Class '{cls}' is empty", None

                # Must NOT contain directories
                for item in contents:
                    if os.path.isdir(os.path.join(cls_path, item)):
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return False, f"Class '{cls}' contains a folder ('{item}') but only image files are allowed", None

    else:
        return False, "Unknown mode", None

    # Move to the Datasets Folder
    final_root = DEST_PATHS[mode]
    os.makedirs(final_root, exist_ok=True)

    final_path = os.path.join(final_root, os.path.basename(tmp_dir))
    if os.path.exists(final_path):
        shutil.rmtree(final_path)

    shutil.move(tmp_dir, final_path)
    os.remove(zip_path)

    return True, "Dataset imported successfully", final_path