import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self, input_size: int, inner_size: int, output_size: int):
        super(DummyModel, self).__init__()
        self.input_layer = nn.Linear(input_size, inner_size)
        self.hidden_layer = nn.Linear(inner_size, inner_size)
        self.output_layer = nn.Linear(inner_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.tanh(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.output_layer(x)

        return x
