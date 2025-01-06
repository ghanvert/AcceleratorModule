import os
import numpy as np
import pandas as pd
from .utils import divide_list

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))

def prepare_list(obj: list):
    if WORLD_SIZE > 1:
        return divide_list(obj, WORLD_SIZE)[RANK]

    return obj

def prepare_dataframe(df: pd.DataFrame):
    if WORLD_SIZE > 1:
        remainder = len(df) % WORLD_SIZE

        partition_size = len(df) // WORLD_SIZE
        start = partition_size * RANK
        end = (partition_size * (RANK + 1)) + (remainder if RANK == WORLD_SIZE - 1 else 0)

        return df.iloc[start:end]
    
    return df

def prepare_array(obj):
    if WORLD_SIZE > 1:
        return np.array_split(obj, WORLD_SIZE)[RANK]
    
    return obj

def prepare(*objs):
    prepared = []
    for obj in objs:
        if isinstance(obj, list):
            prepared.append(prepare_list(obj))
        elif isinstance(obj, pd.DataFrame):
            prepared.append(prepare_dataframe(obj))
        elif hasattr(obj, "__len__"):
            prepared.append(prepare_array(obj))
        else:
            prepared.append(obj)

    return prepared if len(prepared) > 1 else prepared[0]
