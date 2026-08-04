import os
import json
import time
import threading
import subprocess

from utils.models_page import write_results
from utils import hestia_client as hc

# If a "running" job hasn't updated its heartbeat within this many seconds,
# we treat it as dead (e.g. platform/container crashed mid-training).
TRAIN_JOB_HEARTBEAT_INTERVAL = 15
TRAIN_JOB_STALE_TIMEOUT = 90  # 6x the heartbeat interval

def _job_status_path(user_root_fn, user_slug, job_id):
    return os.path.join(user_root_fn(user_slug), "train_jobs", f"{job_id}.json")


def _mark_stale_if_dead(user_root_fn, user_slug, job_id, status):
    """If status is 'running' but its heartbeat is too old, rewrite it as
    'failed' on disk and return the corrected status. Otherwise return as-is."""
    if not status or status.get("status") != "running":
        return status

    last_heartbeat = status.get("last_heartbeat")
    if last_heartbeat is None:
        stale = True
    else:
        stale = (time.time() - last_heartbeat) > TRAIN_JOB_STALE_TIMEOUT

    if stale:
        status["status"] = "failed"
        status["error"] = "Training lost contact with the server (no heartbeat)."
        _write_job_status(user_root_fn, user_slug, job_id, status)

    return status


def _write_job_status(user_root_fn, user_slug, job_id, data):
    path = _job_status_path(user_root_fn, user_slug, job_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(tmp_path, path)


def _read_job_status(user_root_fn, user_slug, job_id):
    path = _job_status_path(user_root_fn, user_slug, job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_training_job(user_root_fn, user_slug, job_id, cmd, mode, paths,
                       hestia_enabled, push_to_hestia_fn, logger):
    prior = _read_job_status(user_root_fn, user_slug, job_id) or {}
    experiment_id = prior.get("experiment_id")
    dataset_id = prior.get("dataset_id")
    owner_email = prior.get("owner_email")

    try:
        proc = subprocess.Popen(cmd)

        while True:
            returncode = proc.poll()
            if returncode is not None:
                break
            prior["status"] = "running"
            prior["last_heartbeat"] = time.time()
            _write_job_status(user_root_fn, user_slug, job_id, prior)
            time.sleep(TRAIN_JOB_HEARTBEAT_INTERVAL)

        process = subprocess.CompletedProcess(cmd, returncode)
        if process.returncode == 0:
            write_results(*paths[mode])
            status = {"status": "succeeded", "return_code": process.returncode}
            if hestia_enabled:
                try:
                    push_to_hestia_fn(
                        user_slug, mode, paths[mode], experiment_id, dataset_id, owner_email)
                except Exception as exc:
                    logger.warning(f"HESTIA model push failed: {exc}")
        else:
            status = {"status": "failed", "return_code": process.returncode}
            if hestia_enabled and experiment_id:
                hc.update_experiment(experiment_id, status="failed",
                                     error=f"training exited with code {process.returncode}")
    except Exception as exc:
        status = {"status": "failed", "return_code": None, "error": str(exc)}
        if hestia_enabled and experiment_id:
            hc.update_experiment(experiment_id, status="failed", error=str(exc))

    for key in ("experiment_id", "dataset_id", "owner_email"):
        if prior.get(key):
            status[key] = prior[key]
    _write_job_status(user_root_fn, user_slug, job_id, status)