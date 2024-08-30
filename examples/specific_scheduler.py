import torch.nn as nn
from accmt import AcceleratorModule, Trainer, HyperParameters
from dummy_model import DummyModel
from dummy_dataset import DummyDataset

class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch):
        x, y = batch 
        x = self.model(x)
        
        loss = self.criterion(x, y)

        return loss

module = DummyModule()

train_dataset = DummyDataset()

trainer = Trainer(hps_config=HyperParameters(epochs=2, batch_size=2), model_path="dummy_model")
trainer.fit(module, train_dataset)
