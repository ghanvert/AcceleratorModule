import os
import numpy as np
import pandas as pd
from .utils import divide_list

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))
MASTER_PROCESS = RANK == 0
LAST_PROCESS = RANK == WORLD_SIZE - 1

def prepare_list(obj: list, even: bool = False):
    if WORLD_SIZE > 1:
        return divide_list(obj, WORLD_SIZE, even=even)[RANK]

    return obj

def prepare_dataframe(df: pd.DataFrame, even: bool = False) -> tuple[pd.DataFrame, int]:
    remainder = 0
    if WORLD_SIZE > 1:
        partition_size, remainder = divmod(len(df), WORLD_SIZE)

        for rank, i in enumerate(range(0, len(df), partition_size + remainder)):
            if rank == RANK:
                df = df.iloc[i:i + partition_size + remainder]
                break

        if even and LAST_PROCESS and remainder != 0:
            last_row = df.iloc[-1:]
            df = pd.concat([df] + [last_row] * (remainder * (WORLD_SIZE - 1)), ignore_index=True)

        remainder = remainder * (WORLD_SIZE - 1)
    
    return df, remainder

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

def divide_into_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def iterbatch(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
