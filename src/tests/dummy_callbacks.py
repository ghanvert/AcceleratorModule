from src.accmt.callbacks import Callback
from src.accmt.decorators import on_main_process

class DummyCallback(Callback):
    @on_main_process
    def on_after_training_step(self):
        print("after_training_step")
