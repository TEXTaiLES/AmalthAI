import os
import re
from utils.user_paths import user_root
import csv
import uuid

from flask import session, url_for

from utils import hestia_client as hc
from utils.user_paths import user_root


HESTIA_ENABLED = False


def init(hestia_enabled):
    global HESTIA_ENABLED
    HESTIA_ENABLED = bool(hestia_enabled)

def _parse_classification_output(output_text):
    """Extract the classifier's selected class and probability table."""
    predicted_class = ""
    confidence = ""
    probabilities = []

    for line in (output_text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        predicted_match = re.match(
            r"Predicted Class:\s*(.*?)\s*(?:\|\s*Confidence:\s*(.*))?$",
            line,
            re.IGNORECASE,
        )
        if predicted_match:
            predicted_class = predicted_match.group(1).strip()
            confidence = (predicted_match.group(2) or "").strip()
            continue

        probability_match = re.match(
            r"([^:]+):\s*.*?\(([0-9]+(?:\.[0-9]+)?)%\)", line
        )
        if probability_match:
            probabilities.append({
                "name": probability_match.group(1).strip(),
                "value": f"{probability_match.group(2)}%",
            })

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }

def _inference_handoff_path(user_slug, relative_path):
    """Resolve a session-provided inference file without allowing path escape."""
    inference_dir = os.path.abspath(os.path.join(user_root(user_slug), "inference"))
    path = os.path.abspath(os.path.join(inference_dir, relative_path))
    if not path.startswith(inference_dir + os.sep):
        return None
    return path


def _inference_params(user_slug):
    return {
        "segmentation": {
            "csv": os.path.join(user_root(user_slug), "models_db", "trained_models_db_segm.csv"),
            "metric": "mIoU Score",
        },
        "detection": {
            "csv": os.path.join(user_root(user_slug), "models_db", "trained_models_db_od.csv"),
            "metric": "mAP 50-95 Score",
        },
        "classification": {
            "csv": os.path.join(user_root(user_slug), "models_db", "trained_models_db_cls.csv"),
            "metric": "Accuracy",
        },
    }


def _resolve_inference_model(mode_params, raw_id):
    """Resolve either a HESTIA model UUID or a legacy CSV row number."""
    model = None
    hestia_model = None
    model_id = raw_id

    if HESTIA_ENABLED:
        hestia_model = hc.get_model(raw_id)
        if hestia_model:
            model = {
                "name": hestia_model.get("name"),
                "trained_on": hestia_model.get("trained_on"),
                "score": hestia_model.get("score"),
                "date": hestia_model.get("trained_date"),
                "model_id": hestia_model.get("model_id"),
            }
            model_id = hestia_model.get("model_id")

    if model is None:
        try:
            model_id = int(raw_id)
        except (TypeError, ValueError):
            return None, None, None

        try:
            with open(mode_params["csv"], newline="", encoding="utf-8") as csvfile:
                models = list(csv.DictReader(csvfile))
        except OSError:
            return None, None, None

        model = next((row for index, row in enumerate(models, 1) if index == model_id), None)
        if not model:
            return None, None, None

    return model, hestia_model, model_id


def _list_inference_images(folder):
    if not os.path.isdir(folder):
        return []

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, filename))
        and os.path.splitext(filename)[1].lower() in image_extensions
    )


def _load_segmentation_color_table(user_slug, dataset_name):
    if not dataset_name:
        return []

    labelmap_path = os.path.join(
        user_root(user_slug), "Datasets", "Segmentation", dataset_name, "labelmap.txt"
    )
    if not os.path.isfile(labelmap_path):
        return []

    color_table = []
    with open(labelmap_path, "r", encoding="utf-8") as labelmap:
        for line in labelmap:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            try:
                red, green, blue = map(int, parts[1].split(","))
            except ValueError:
                continue
            color_table.append({
                "label": parts[0],
                "rgb": f"rgb({red},{green},{blue})",
                "hex": "#{:02x}{:02x}{:02x}".format(red, green, blue),
            })
    return color_table


def _collect_inference_runs(user_slug, mode, model_id, inference_root=None,
                            run_ids=None, url_prefix=None):
    """Rebuild completed inference runs from a local or HESTIA cache root."""
    inference_root = inference_root or os.path.join(user_root(user_slug), "inference", mode)
    url_prefix = url_prefix or mode
    inputs_root = os.path.join(inference_root, "inputs", str(model_id))
    outputs_root = os.path.join(inference_root, "outputs", str(model_id))
    if not os.path.isdir(inputs_root):
        return []

    runs = []
    for timestamp in sorted(os.listdir(inputs_root), reverse=True):
        if run_ids is not None and timestamp not in run_ids:
            continue
        input_dir = os.path.join(inputs_root, timestamp)
        output_dir = os.path.join(outputs_root, timestamp)
        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            continue

        run_results = []
        for input_path in _list_inference_images(input_dir):
            filename = os.path.basename(input_path)
            base_name = os.path.splitext(filename)[0]

            if mode == "classification":
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                if not os.path.isfile(output_path):
                    continue
                with open(output_path, "r", encoding="utf-8") as output_file:
                    output_text = output_file.read()

                gradcam_filename = f"{base_name}_gradcam.jpg"
                gradcam_path = os.path.join(output_dir, gradcam_filename)
                run_results.append({
                    "input_file": url_for(
                        "user_inference_files",
                        filename=f"{url_prefix}/inputs/{model_id}/{timestamp}/{filename}",
                    ),
                    "output_text": output_text,
                    "gradcam_file": (
                        url_for(
                            "user_inference_files",
                            filename=f"{url_prefix}/outputs/{model_id}/{timestamp}/{gradcam_filename}",
                        )
                        if os.path.isfile(gradcam_path) else None
                    ),
                    "filename": filename,
                    "timestamp": timestamp,
                    "result_root": url_prefix,
                })
                continue

            output_search_dir = os.path.join(output_dir, "predict") if mode == "detection" else output_dir
            output_files = _list_inference_images(output_search_dir)
            output_by_base = {
                os.path.splitext(os.path.basename(output_path))[0]: os.path.basename(output_path)
                for output_path in output_files
            }
            output_filename = output_by_base.get(base_name)
            if not output_filename and mode == "segmentation":
                output_filename = next(
                    (
                        name for output_base, name in output_by_base.items()
                        if base_name in output_base or output_base in base_name
                    ),
                    None,
                )
            if not output_filename:
                continue

            output_segment = "predict/" if mode == "detection" else ""
            run_results.append({
                "input_image": url_for(
                    "user_inference_files",
                    filename=f"{url_prefix}/inputs/{model_id}/{timestamp}/{filename}",
                ),
                "output_image": url_for(
                    "user_inference_files",
                    filename=f"{url_prefix}/outputs/{model_id}/{timestamp}/{output_segment}{output_filename}",
                ),
            })

        if run_results:
            runs.append({"timestamp": timestamp, "results": run_results})

    return runs


def _collect_available_inference_runs(user_slug, mode, model_id):
    """Use HESTIA as authoritative source, falling back locally if unavailable."""
    if HESTIA_ENABLED:
        hestia_rows = hc.list_inference_runs(user_slug, mode, model_id)
        if hestia_rows is not None:
            cache_root = os.path.join(
                user_root(user_slug), "inference", "_hestia_results", mode
            )
            available_ids = set()
            for row in hestia_rows:
                if row.get("status") != "completed":
                    continue
                if hc.ensure_inference_results_local(row, cache_root):
                    available_ids.add(str(row.get("inference_id")))
            cached_runs = _collect_inference_runs(
                user_slug,
                mode,
                model_id,
                inference_root=cache_root,
                run_ids=available_ids,
                url_prefix=f"_hestia_results/{mode}",
            )
            cached_by_id = {run["timestamp"]: run for run in cached_runs}
            ordered_runs = []
            for row in hestia_rows:
                inference_id = str(row.get("inference_id"))
                cached_run = cached_by_id.get(inference_id)
                if cached_run is None:
                    continue
                cached_run["timestamp"] = row.get("created_at") or inference_id
                ordered_runs.append(cached_run)
            return ordered_runs, hestia_rows

    return _collect_inference_runs(user_slug, mode, model_id), None


def _has_available_inference_results(user_slug, mode, model_id):
    """Check result availability without downloading inference artifacts."""
    if HESTIA_ENABLED:
        rows = hc.list_inference_runs(user_slug, mode, model_id)
        if rows is not None:
            return any(
                row.get("status") == "completed"
                and bool(row.get("inputs"))
                and bool(row.get("outputs"))
                for row in rows
            )
    return bool(_collect_inference_runs(user_slug, mode, model_id))


def _store_vlm_handoff(input_path, gradcam_path, output_path):
    """Store a small, bounded saved-file handoff in the signed user session."""
    handoff_id = uuid.uuid4().hex
    handoffs = session.get("vlm_handoffs", {})
    handoffs[handoff_id] = {
        "input_path": input_path,
        "gradcam_path": gradcam_path,
        "output_path": output_path,
    }
    session["vlm_handoffs"] = dict(list(handoffs.items())[-20:])
    session.modified = True
    return handoff_id