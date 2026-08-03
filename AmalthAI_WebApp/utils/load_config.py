import yaml
from pathlib import Path

def load_config(path):
    config = {}

    for file in [path, "config.override.yml"]:
        if Path(file).exists():
            with open(file, "r") as f:
                config.update(yaml.safe_load(f))

    return config

def chown_target(config):
    permissions = config.get("permissions", {})
    uid = permissions.get("chown_uid")
    gid = permissions.get("chown_gid")
    return f"{int(uid)}:{int(gid)}"