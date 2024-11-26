import torch
import torch.nn as nn
from src.accmt import AcceleratorModule, Trainer, HyperParameters, Optimizer, Scheduler, Monitor, set_seed, accelerator
from src.accmt.tracker import MLFlow
from .dummy_model import DummyModel
from .dummy_dataset import DummyDataset
from .dummy_metrics import Accuracy

set_seed(42)

class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch 
        x = self.model(x)
        
        loss = self.criterion(x, y)

        return loss
    
    def validation_step(self, batch):
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        predictions = torch.argmax(x, dim=1)
        references = torch.argmax(y, dim=1)

        return {
            "loss": loss,
            "accuracy": (predictions, references),
            "my_own_metric": (predictions, references)
        }

module = DummyModule()

train_dataset = DummyDataset()
val_dataset = DummyDataset()

metrics = [Accuracy("accuracy")]
trainer = Trainer(
    hps_config=HyperParameters(
        epochs=2,
        batch_size=(2, 1, 1),
        optim=Optimizer.AdamW,
        optim_kwargs={"lr": 0.001, "weight_decay": 0.01},
        scheduler=Scheduler.LinearWithWarmup,
        scheduler_kwargs={"warmup_ratio": 0.03}
    ),
    model_path="dummy_model",
    track_name="Dummy training",
    run_name="dummy_run",
    model_saving=["accuracy"],
    evaluate_every_n_steps=1,
    checkpoint_every="eval",
    logging_dir="localhost:5075",
    log_with=MLFlow,
    log_every=2,
    monitor=Monitor(grad_norm=True),
    compile=True,
    dataloader_num_workers=accelerator.num_processes,
    eval_when_start=True,
    metrics=metrics
)

if __name__ == "__main__":
    trainer.fit(module, train_dataset, val_dataset)
