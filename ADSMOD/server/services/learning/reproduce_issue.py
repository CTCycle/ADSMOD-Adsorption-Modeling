
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ADSMOD.server.services.learning.device import DeviceDataLoader

# Mocking TorchDictDataset from loader.py
class TorchDictDataset(Dataset):
    def __init__(self):
        self.length = 10
        self.inputs = {"a": torch.randn(10, 5), "b": torch.randn(10, 5)}
        self.outputs = torch.randn(10, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inputs = {key: value[idx] for key, value in self.inputs.items()}
        return inputs, self.outputs[idx]

def test():
    dataset = TorchDictDataset()
    # Check dataset item type
    print(f"Dataset item type: {type(dataset[0])}")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # Check loader batch type
    batch = next(iter(loader))
    print(f"DataLoader batch type: {type(batch)}")
    
    device = torch.device("cpu")
    device_loader = DeviceDataLoader(loader, device)
    
    # Check DeviceDataLoader batch type
    device_batch = next(iter(device_loader))
    print(f"DeviceDataLoader batch type: {type(device_batch)}")
    print(f"DeviceDataLoader batch content: {device_batch}")

if __name__ == "__main__":
    test()
