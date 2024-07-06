import json
import yaml


def read(file: str) -> dict:
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    data["version"] = file.split("/")[-1].removesuffix(".yaml").removesuffix(".yml")
    return data


def save_status(status: dict, to: str):
    json_string = json.dumps(status, indent=4)
    with open(to, "w") as f:
        f.write(json_string)


def read_status(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    return data
