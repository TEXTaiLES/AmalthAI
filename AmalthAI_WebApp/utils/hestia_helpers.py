import os
import csv
from werkzeug.utils import secure_filename

from utils import hestia_client as hc
from utils.helpers import load_models
from utils.user_paths import get_current_user_email

HESTIA_METRIC = None


def init(hestia_metric):
    global HESTIA_METRIC
    HESTIA_METRIC = hestia_metric


def _dataset_manifest(mode, final_path, num_classes):
    """Best-effort, mode-aware manifest describing a validated dataset directory."""
    manifest = {"mode": mode, "num_classes": num_classes, "archive_format": "tar.gz"}

    def _count(*parts):
        path = os.path.join(final_path, *parts)
        return len(os.listdir(path)) if os.path.isdir(path) else 0

    if mode == "segmentation":
        labelmap = os.path.join(final_path, "labelmap.txt")
        if os.path.exists(labelmap):
            with open(labelmap, "r", encoding="utf-8") as f:
                manifest["labelmap"] = [ln.strip() for ln in f if ln.strip()]
        manifest["counts"] = {"images_train": _count("images", "train"),
                              "images_val": _count("images", "val")}
    elif mode == "detection":
        manifest["counts"] = {"train_images": _count("train", "images"),
                              "valid_images": _count("valid", "images")}
    elif mode == "classification":
        train_dir = os.path.join(final_path, "train")
        root = train_dir if os.path.isdir(train_dir) else final_path
        classes = ([d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))]
                   if os.path.isdir(root) else [])
        manifest["classes"] = sorted(classes)
    return manifest


def _persist_dataset_to_hestia(user_slug, mode, final_path, num_classes, form):
    """Push a validated dataset directory to HESTIA (idempotent, non-fatal)."""
    name = os.path.basename(os.path.normpath(final_path))
    links = {
        "scan_id": form.get("linked_scan_id"),
        "artifact_id": form.get("linked_artifact_id"),
        "reconstruction_id": form.get("linked_reconstruction_id"),
    }
    links = {k: v for k, v in links.items() if v}
    hc.upload_dataset(
        owner_slug=user_slug,
        mode=mode,
        name=name,
        src_dir=final_path,
        owner_email=get_current_user_email(),
        num_classes=int(num_classes) if num_classes else None,
        manifest=_dataset_manifest(mode, final_path, num_classes),
        links=links or None,
    )


def _push_trained_model_to_hestia(user_slug, mode, mode_paths, experiment_id,
                                  dataset_id, owner_email):
    """After a successful run, read the freshly-appended CSV row and push the model."""
    db_loc = mode_paths[1]  # trained_models_db_<mode>.csv
    with open(db_loc, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    try:
        score = float(row.get("score"))
    except (TypeError, ValueError):
        score = None

    model_id = hc.push_model(
        owner_slug=user_slug,
        mode=mode,
        name=row.get("name"),
        trained_on=row.get("trained_on"),
        weights_path=row.get("checkpoint_path"),
        config_path=row.get("config_path"),
        owner_email=owner_email,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        score=score,
        metric_name=HESTIA_METRIC.get(mode),
        trained_date=row.get("date"),
    )
    if experiment_id:
        hc.update_experiment(
            experiment_id, status="succeeded", result_model_id=model_id,
            metrics={"score": score, "name": row.get("name"),
                     "trained_on": row.get("trained_on")},
        )
    return model_id


def _hestia_models_for_template(user_slug, mode, csv_path):
    """Trained models for the UI: from HESTIA when reachable, else the local CSV.

    Maps HESTIA rows to the shape models.html / inference expect, using the UUID
    model_id as the link id (id=...) so inference selects by UUID, not row index.
    """
    rows = hc.list_models(user_slug, mode)
    if rows is None:  # HESTIA unreachable -> fall back to the local CSV
        return load_models(csv_path)
    return [
        {
            "id"        : r.get("model_id"),
            "model_id"  : r.get("model_id"),
            "name"      : r.get("name"),
            "trained_on": r.get("trained_on"),
            "score"     : r.get("score"),
            "date"      : r.get("trained_date"),
        }
        for r in rows
    ]


def _persist_inference_to_hestia(user_slug, mode, model_id, model_name, dataset_name,
                                 files, save_dir, output_map, color_table):
    """Record an inference run in HESTIA and upload its input + output images."""
    inf_id = hc.create_inference_run(user_slug, mode, model_id=model_id,
                                     model_name=model_name, dataset_name=dataset_name)
    if not inf_id:
        return
    input_paths = [os.path.join(save_dir, secure_filename(f.filename))
                   for f in files if f.filename]
    hc.upload_inference_inputs(inf_id, input_paths)

    # mapping: output filename -> originating input filename
    mapping = {}
    for f in files:
        if not f.filename:
            continue
        fn = secure_filename(f.filename)
        base = os.path.splitext(fn)[0]
        for out_name in output_map:
            if base in out_name:
                mapping[out_name] = fn
                break
    out_paths = list(output_map.values())
    hc.upload_inference_outputs(inf_id, out_paths, mapping=mapping,
                               color_table=color_table or None)