import yaml

def read(file: str) -> dict:
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    data["version"] = file.split("/")[-1].removesuffix(".yaml")
    return data
