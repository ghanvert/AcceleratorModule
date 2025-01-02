import os
import numpy as np
from .utils import divide_list

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))

def prepare_list(obj: list):
    if WORLD_SIZE > 1:
        return divide_list(obj)[RANK]

    return obj

def prepare_dataframe(obj):
    return np.array_split(obj, WORLD_SIZE)[RANK]

def prepare_array(obj):
    return np.array_split(obj, WORLD_SIZE)[RANK]

def prepare(*objs):
    prepared = []
    for obj in objs:
        if isinstance(obj, list):
            prepared.append(prepare_list(obj))
        elif hasattr(obj, "__len__"):
            prepared.append(prepare_array(obj))
        else:
            prepared.append(obj)

    return prepared if len(prepared) > 1 else prepared[0]
