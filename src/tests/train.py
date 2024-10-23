import torch
import torch.nn as nn
from src.accmt import AcceleratorModule, Trainer, HyperParameters, Optimizer, Scheduler, Monitor, set_seed, accelerator
from src.accmt.tracker import MLFlow
from .dummy_model import DummyModel
from .dummy_dataset import DummyDataset

set_seed(42)

class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch):
        x, y = batch 
        x = self.model(x)
        
        loss = self.criterion(x, y)

        return loss
    
    def test_step(self, batch):
        x, y = batch
        x = self.model(x)

        predictions = torch.argmax(x, dim=1)
        references = torch.argmax(y, dim=1)

        return {
            "accuracy": (predictions, references),
            "my_own_metric": (predictions, references)
        }
    
    def compute_metrics(self, predictions, references):
        # ... do some operations to get the metric value

        return {
            "my_own_metric": -self.status_dict["global_step"]
        }

module = DummyModule()

train_dataset = DummyDataset()
val_dataset = DummyDataset()
test_dataset = DummyDataset()

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
    resume=False,
    model_saving=["best_accuracy", "best_my_own_metric"],
    evaluate_every_n_steps=1,
    checkpoint_every="eval",
    logging_dir="localhost:5075",
    log_with=MLFlow,
    log_every=2,
    monitor=Monitor(grad_norm=True),
    additional_metrics=["accuracy", "my_own_metric"],
    additional_metrics_comparators={
        "my_own_metric": "<"
    },
    compile=True,
    dataloader_num_workers=accelerator.num_processes,
    eval_when_start=True
)

if __name__ == "__main__":
    trainer.fit(module, train_dataset, val_dataset, test_dataset)
