import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def chown_target(config):
    permissions = config.get("permissions", {})
    uid = permissions.get("chown_uid")
    gid = permissions.get("chown_gid")
    return f"{int(uid)}:{int(gid)}"