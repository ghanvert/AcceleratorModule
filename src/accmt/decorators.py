from . import accelerator

def on_main_process(function):
    return accelerator.on_main_process(function)

def on_last_process(function):
    return accelerator.on_last_process(function)

def on_local_main_process(function):
    return accelerator.on_local_main_process(function)

def on_local_process(function, local_process_index: int):
    return accelerator.on_local_process(function, local_process_index)

def on_process(function, process_index: int):
    return accelerator.on_process(function, process_index)
