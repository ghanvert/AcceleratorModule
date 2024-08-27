import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self):
        self.dataset = [
            ([3, 2], [1, 0, 0]),
            ([2, 1], [0, 0, 1]),
            ([2, 0], [0, 1, 0]),
            ([4, 3], [1, 0, 0])
        ]
        
    def __getitem__(self, idx):
        input = torch.tensor(self.dataset[idx][0], dtype=torch.float32)
        target = torch.tensor(self.dataset[idx][1], dtype=torch.float32)

        return input, target
    
    def __len__(self):
        return len(self.dataset)
