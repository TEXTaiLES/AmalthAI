"""HESTIA data-lake client for AmalthAI.

Thin wrapper over HESTIA's ``/amalthai/*`` REST endpoints. Mirrors the existing
``requests`` usage in app.py (explicit timeouts, ``resp.ok`` checks). Every call
is wrapped so that a HESTIA outage is **non-fatal**: functions return
``None``/``False``/``[]`` and log, so AmalthAI's training/inference never breaks
if the data lake is unreachable (the CSV path remains the fallback).

Configuration (env, with config.yml fallback via ``configure()``):
  - HESTIA_BASE_URL  e.g. http://api:5000 (co-located) or https://api.textailes.athenarc.gr
  - HESTIA_API_KEY   the shared HESTIA API key (Bearer)

Ownership: every record is scoped by ``owner_slug`` (the AmalthAI user slug).
"""
import os
import json
import gzip
import hashlib
import logging
import tarfile
import tempfile

import requests

logger = logging.getLogger(__name__)

# Timeouts: short for metadata calls, long for blob upload/download.
T = 30
T_LONG = 1200

_DEFAULT_BASE_URL = "https://api.textailes.athenarc.gr"


# --------------------------------------------------------------------------- #
# Configuration (read lazily so env set after import still applies)
# --------------------------------------------------------------------------- #
def configure(base_url=None, api_key=None):
    """Fill base URL / key from config.yml *without* overriding real env vars.

    Env (set by docker-compose in production) always wins; this only provides
    fallbacks for bare-metal runs that pass values from config.yml.
    """
    if base_url:
        os.environ.setdefault("HESTIA_BASE_URL", base_url)
    if api_key:
        os.environ.setdefault("HESTIA_API_KEY", api_key)


def _base():
    return os.environ.get("HESTIA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _headers():
    return {"Authorization": f"Bearer {os.environ.get('HESTIA_API_KEY', '')}"}


def _url(path):
    return f"{_base()}{path}"


# --------------------------------------------------------------------------- #
# Low-level HTTP helpers
# --------------------------------------------------------------------------- #
def _get(path, params=None):
    try:
        r = requests.get(_url(path), headers=_headers(), params=params, timeout=T)
        if r.ok:
            return r.json()
        logger.error(f"[hestia] GET {path} -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logger.error(f"[hestia] GET {path} error: {e}")
    return None


def _post_json(path, payload):
    try:
        r = requests.post(_url(path),
                          headers={**_headers(), "Content-Type": "application/json"},
                          json=payload, timeout=T)
        if r.ok:
            return r.json() if r.content else {}
        logger.error(f"[hestia] POST {path} -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logger.error(f"[hestia] POST {path} error: {e}")
    return None


def _patch_json(path, payload):
    try:
        r = requests.patch(_url(path),
                           headers={**_headers(), "Content-Type": "application/json"},
                           json=payload, timeout=T)
        if r.ok:
            return r.json() if r.content else {}
        logger.error(f"[hestia] PATCH {path} -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logger.error(f"[hestia] PATCH {path} error: {e}")
    return None


def _post_file(path, file_path, data=None, field="file", filename=None):
    """POST a single file (multipart), streamed from disk."""
    try:
        with open(file_path, "rb") as fh:
            files = {field: (filename or os.path.basename(file_path), fh,
                             "application/octet-stream")}
            r = requests.post(_url(path), headers=_headers(), files=files,
                              data=data or {}, timeout=T_LONG)
        if r.ok:
            return r.json() if r.content else {}
        logger.error(f"[hestia] POST(file) {path} -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logger.error(f"[hestia] POST(file) {path} error: {e}")
    return None


def _post_files_multi(path, file_paths, data=None, field="file"):
    """POST several files under the same field name (multipart), streamed."""
    handles = []
    try:
        files = []
        for fp in file_paths:
            fh = open(fp, "rb")
            handles.append(fh)
            files.append((field, (os.path.basename(fp), fh, "application/octet-stream")))
        r = requests.post(_url(path), headers=_headers(), files=files,
                          data=data or {}, timeout=T_LONG)
        if r.ok:
            return r.json() if r.content else {}
        logger.error(f"[hestia] POST(files) {path} -> {r.status_code}: {r.text[:300]}")
        return None
    except Exception as e:
        logger.error(f"[hestia] POST(files) {path} error: {e}")
        return None
    finally:
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass


def _download(path, dest_path):
    """Stream a download to dest_path. Returns True on success."""
    try:
        parent = os.path.dirname(dest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with requests.get(_url(path), headers=_headers(), stream=True, timeout=T_LONG) as r:
            if not r.ok:
                logger.error(f"[hestia] GET(download) {path} -> {r.status_code}")
                return False
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"[hestia] download {path} error: {e}")
        return False


# --------------------------------------------------------------------------- #
# Archive + hashing utilities
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_deterministic(tar, src_dir, arcname):
    """Add src_dir to tar with normalized metadata so identical content hashes equal."""
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()
        for fn in sorted(files):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src_dir)
            ti = tar.gettarinfo(full, arcname=os.path.join(arcname, rel))
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(full, "rb") as fh:
                tar.addfile(ti, fh)


def make_dataset_archive(src_dir, dest_path, arcname=None):
    """Create a deterministic .tar.gz of src_dir (top-level folder = arcname)."""
    arcname = arcname or os.path.basename(os.path.normpath(src_dir))
    # filename="" stops gzip from embedding the (random temp) source filename in
    # its header, which would otherwise make the hash non-deterministic.
    with open(dest_path, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            _add_deterministic(tar, src_dir, arcname)
    return dest_path


def extract_archive(archive_path, dest_root):
    os.makedirs(dest_root, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        try:
            tar.extractall(dest_root, filter="data")  # py3.12+
        except TypeError:
            tar.extractall(dest_root)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def find_dataset(owner_slug, mode, name):
    rows = _get("/amalthai/datasets",
                {"owner_slug": owner_slug, "mode": mode, "name": name})
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def create_dataset(owner_slug, mode, name, owner_email=None, num_classes=None,
                   manifest=None, content_hash=None, links=None):
    payload = {
        "owner_slug": owner_slug, "owner_email": owner_email,
        "name": name, "mode": mode, "num_classes": num_classes,
        "manifest": manifest, "content_hash": content_hash,
    }
    if links:
        payload.update({
            "linked_scan_id": links.get("scan_id"),
            "linked_artifact_id": links.get("artifact_id"),
            "linked_reconstruction_id": links.get("reconstruction_id"),
        })
    return _post_json("/amalthai/datasets", payload)


def upload_dataset_archive(dataset_id, archive_path, content_hash=None):
    data = {"content_hash": content_hash} if content_hash else None
    return _post_file(f"/amalthai/datasets/{dataset_id}/archive", archive_path, data=data)


def download_dataset_archive(dataset_id, dest_path):
    return _download(f"/amalthai/datasets/{dataset_id}/archive", dest_path)


def upload_dataset(owner_slug, mode, name, src_dir, owner_email=None,
                   num_classes=None, manifest=None, links=None):
    """Idempotently push a validated dataset directory to HESTIA.

    Archives src_dir (top-level folder == name), and skips the upload if HESTIA
    already holds an identical (content-hash) copy. Returns the dataset_id.
    """
    archive_path = None
    try:
        fd, archive_path = tempfile.mkstemp(suffix=".tar.gz", prefix=f"{name}_")
        os.close(fd)
        make_dataset_archive(src_dir, archive_path, arcname=name)
        content_hash = sha256_file(archive_path)

        existing = find_dataset(owner_slug, mode, name)
        if existing and existing.get("content_hash") == content_hash and existing.get("object_key"):
            logger.info(f"[hestia] dataset '{name}' unchanged (hash match); skipping upload")
            return existing.get("dataset_id")

        created = create_dataset(owner_slug, mode, name, owner_email=owner_email,
                                 num_classes=num_classes, manifest=manifest,
                                 content_hash=content_hash, links=links)
        if not created:
            return None
        dataset_id = created.get("dataset_id")
        if not upload_dataset_archive(dataset_id, archive_path, content_hash):
            return None
        return dataset_id
    except Exception as e:
        logger.error(f"[hestia] upload_dataset failed for '{name}': {e}")
        return None
    finally:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)


def ensure_dataset_local(owner_slug, mode, name, dest_root):
    """Make sure dest_root/name exists locally, downloading from HESTIA if missing.

    dest_root is the AmalthAI Datasets/<ModeDir> directory; the archive's
    top-level folder is `name`, so extracting into dest_root yields dest_root/name.
    Returns True if the dataset is present locally afterwards.
    """
    local_dir = os.path.join(dest_root, name)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return True  # cache hit
    ds = find_dataset(owner_slug, mode, name)
    if not ds or not ds.get("object_key"):
        logger.warning(f"[hestia] no archived dataset to rehydrate: {owner_slug}/{mode}/{name}")
        return False
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        if not download_dataset_archive(ds["dataset_id"], tmp):
            return False
        extract_archive(tmp, dest_root)
        return os.path.isdir(local_dir)
    except Exception as e:
        logger.error(f"[hestia] ensure_dataset_local failed for '{name}': {e}")
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def list_models(owner_slug, mode=None):
    """Return model rows, or None if HESTIA is unreachable (vs [] when empty)."""
    params = {"owner_slug": owner_slug}
    if mode:
        params["mode"] = mode
    rows = _get("/amalthai/models", params)
    return rows if isinstance(rows, list) else None


def get_model(model_id):
    return _get(f"/amalthai/models/{model_id}", None)


def find_model(owner_slug, mode, name=None, trained_on=None, content_hash=None):
    for row in (list_models(owner_slug, mode) or []):
        if name is not None and row.get("name") != name:
            continue
        if trained_on is not None and row.get("trained_on") != trained_on:
            continue
        if content_hash is not None and row.get("content_hash") != content_hash:
            continue
        return row
    return None


def register_model(owner_slug, mode, name, trained_on=None, owner_email=None,
                   dataset_id=None, experiment_id=None, score=None, metric_name=None,
                   trained_date=None, content_hash=None, extra=None):
    payload = {
        "owner_slug": owner_slug, "owner_email": owner_email, "name": name, "mode": mode,
        "trained_on": trained_on, "dataset_id": dataset_id, "experiment_id": experiment_id,
        "score": score, "metric_name": metric_name, "trained_date": trained_date,
        "content_hash": content_hash, "extra": extra,
    }
    return _post_json("/amalthai/models", payload)


def upload_model_weights(model_id, weights_path, content_hash=None):
    data = {"content_hash": content_hash} if content_hash else None
    return _post_file(f"/amalthai/models/{model_id}/weights", weights_path, data=data)


def upload_model_config(model_id, config_path):
    return _post_file(f"/amalthai/models/{model_id}/config", config_path)


def download_model_weights(model_id, dest_path):
    return _download(f"/amalthai/models/{model_id}/weights", dest_path)


def download_model_config(model_id, dest_path):
    return _download(f"/amalthai/models/{model_id}/config", dest_path)


def push_model(owner_slug, mode, name, trained_on, weights_path, config_path,
               owner_email=None, dataset_id=None, experiment_id=None, score=None,
               metric_name=None, trained_date=None, extra=None):
    """Register a trained model and upload its weights + config. Returns model_id.

    Idempotent: if an identical (name, trained_on, content_hash) model already
    has weights in HESTIA, returns that model_id without re-uploading.
    """
    try:
        content_hash = (sha256_file(weights_path)
                        if weights_path and os.path.exists(weights_path) else None)
        if content_hash:
            existing = find_model(owner_slug, mode, name=name,
                                  trained_on=trained_on, content_hash=content_hash)
            if existing and existing.get("weights_key"):
                logger.info(f"[hestia] model '{name}'/'{trained_on}' unchanged; skipping")
                return existing.get("model_id")

        created = register_model(owner_slug, mode, name, trained_on=trained_on,
                                 owner_email=owner_email, dataset_id=dataset_id,
                                 experiment_id=experiment_id, score=score,
                                 metric_name=metric_name, trained_date=trained_date,
                                 content_hash=content_hash, extra=extra)
        if not created:
            return None
        model_id = created.get("model_id")
        if weights_path and os.path.exists(weights_path):
            upload_model_weights(model_id, weights_path, content_hash)
        if config_path and os.path.exists(config_path):
            upload_model_config(model_id, config_path)
        return model_id
    except Exception as e:
        logger.error(f"[hestia] push_model failed for '{name}': {e}")
        return None


def ensure_model_local(model_row, cache_dir):
    """Ensure a model's weights + config exist under cache_dir, downloading if missing.

    Returns (checkpoint_path, config_path) as local paths usable by the inference
    container (cache_dir lives under /data/<slug>/..., visible to the container).
    """
    os.makedirs(cache_dir, exist_ok=True)
    model_id = model_row.get("model_id")
    weights_key = model_row.get("weights_key")
    config_key = model_row.get("config_key")
    ckpt = cfg = None

    if weights_key:
        ckpt = os.path.join(cache_dir, os.path.basename(weights_key))
        if not (os.path.exists(ckpt) and os.path.getsize(ckpt) > 0):
            if not download_model_weights(model_id, ckpt):
                ckpt = None
    if config_key:
        cfg = os.path.join(cache_dir, os.path.basename(config_key))
        if not (os.path.exists(cfg) and os.path.getsize(cfg) > 0):
            if not download_model_config(model_id, cfg):
                cfg = None
    return ckpt, cfg


# --------------------------------------------------------------------------- #
# Experiments (training runs)
# --------------------------------------------------------------------------- #
def create_experiment(owner_slug, mode, dataset_id=None, dataset_name=None,
                      requested_model=None, params=None, job_id=None, owner_email=None):
    payload = {
        "owner_slug": owner_slug, "owner_email": owner_email, "mode": mode,
        "dataset_id": dataset_id, "dataset_name": dataset_name,
        "requested_model": requested_model, "params": params, "job_id": job_id,
    }
    resp = _post_json("/amalthai/experiments", payload)
    return resp.get("experiment_id") if resp else None


def update_experiment(experiment_id, **fields):
    if not experiment_id:
        return False
    return bool(_patch_json(f"/amalthai/experiments/{experiment_id}", fields))


# --------------------------------------------------------------------------- #
# Inference runs
# --------------------------------------------------------------------------- #
def create_inference_run(owner_slug, mode, model_id=None, model_name=None,
                         dataset_name=None, owner_email=None):
    payload = {
        "owner_slug": owner_slug, "owner_email": owner_email, "mode": mode,
        "model_id": model_id, "model_name": model_name, "dataset_name": dataset_name,
    }
    resp = _post_json("/amalthai/inference-runs", payload)
    return resp.get("inference_id") if resp else None


def upload_inference_inputs(inference_id, file_paths):
    resp = _post_files_multi(f"/amalthai/inference-runs/{inference_id}/inputs", file_paths)
    return resp.get("inputs") if resp else None


def upload_inference_outputs(inference_id, file_paths, mapping=None, color_table=None):
    """mapping: {output_filename: input_filename}; color_table: list/dict (segmentation)."""
    data = {}
    if mapping is not None:
        data["mapping"] = json.dumps(mapping)
    if color_table is not None:
        data["color_table"] = json.dumps(color_table)
    resp = _post_files_multi(f"/amalthai/inference-runs/{inference_id}/outputs",
                             file_paths, data=data)
    return resp.get("outputs") if resp else None
