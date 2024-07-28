import inspect
import re
import datetime

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

def get_num_required_params(fn):
    parameters = inspect.signature(fn).parameters
    return len([p for p in parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        combined.update(d)
    return combined

time_prefix = lambda: "["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]+"]"
