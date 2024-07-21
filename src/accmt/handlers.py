from dataclasses import dataclass

@dataclass
class Handler:
    KEYBOARD = 1
    CUDA_OUT_OF_MEMORY = 2
    ANY = 3
