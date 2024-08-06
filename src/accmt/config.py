import json
import yaml


def read(file: str) -> dict:
    data = yaml.safe_load(open(file))
    data["version"] = file.split("/")[-1].removesuffix(".yaml").removesuffix(".yml")
    return data


def save_status(status: dict, to: str):
    json_string = json.dumps(status, indent=4)
    open(to, "w").write(json_string)


def read_status(path: str) -> dict:
    return json.load(open(path))
