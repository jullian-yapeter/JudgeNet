import json


def write_json(obj, dest_path):
    with open(dest_path, "w") as F:
        json.dump(obj, F)
