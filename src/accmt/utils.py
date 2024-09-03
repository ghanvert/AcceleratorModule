import re
import datetime
import pandas as pd
import json
import sys
import os
import warnings
from contextlib import contextmanager

units = {
    "epoch": {"epoch", "ep", "epochs", "eps"},
    "step": {"step", "st", "steps", "sts"},
    "eval": {"evaluation", "eval", "evaluations", "evals"}
}

def is_url(string):
    if string in ["localhost", "127.0.0.1"]:
        return True
    
    url_regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' 
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_regex, string) is not None

def get_number_and_unit(string: str):
    match = re.match(r"(\d+)(\D+)", string)

    if match:
        number = int(match.group(1))
        text = match.group(2).strip().lower()
    else:
        number = 1
        text = string.strip().lower()

    unit = None
    for k, v in units.items():
        if text in v:
            unit = k
            break

    return number, unit

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        combined.update(d)
    return combined

def divide_list(lst: list, parts: int):
    k, m = divmod(len(lst), parts)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(parts)]

time_prefix = lambda: "["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]+"]"

PANDAS_READER_MAP = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xml": pd.read_xml,
    "feather": pd.read_feather,
    "parquet": pd.read_parquet,
    "pickle": pd.read_pickle,
    "pkl": pd.read_pickle
}

def save_status(status: dict, to: str):
    json_string = json.dumps(status, indent=4)
    open(to, "w").write(json_string)


def read_status(path: str) -> dict:
    return json.load(open(path))

@contextmanager
def suppress_print_and_warnings(verbose=False):
    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
    else:
        yield
