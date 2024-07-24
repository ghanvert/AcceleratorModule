from dataclasses import dataclass

@dataclass
class Handler:
    """
    Properties:
        `KEYBOARD`: Handles `KeyboardInterrupt` exceptions (CTRL + C).
        `CUDA_OUT_OF_MEMORY`: Handles cuda out of memory errors derived from `RuntimeError` exceptions.
        `ANY`: Handles any other exception not considered in these properties.
        `ALL`: Handles all possible exceptions.
    """
    KEYBOARD            = 1
    CUDA_OUT_OF_MEMORY  = 2
    ANY                 = 3
    ALL                 = 4
