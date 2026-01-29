
import torch
from torch.utils.data import DataLoader, Dataset
from ADSMOD.server.utils.learning.device import DeviceDataLoader

class MockDataset(Dataset):
    def __len__(self):
        return 5
    def __getitem__(self, idx):
        return {"input": torch.tensor([idx])}, torch.tensor([idx])

def test_device_data_loader():
    dataset = MockDataset()
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device("cpu")
    
    device_loader = DeviceDataLoader(loader, device)
    
    print("Testing DeviceDataLoader yielded type...")
    for batch in device_loader:
        if isinstance(batch, tuple):
            print("SUCCESS: Yielded a tuple.")
            print(f"Structure: {type(batch)} of length {len(batch)}")
            if isinstance(batch[0], dict) and isinstance(batch[1], torch.Tensor):
                 print("Content types match expected (dict, Tensor).")
            else:
                 print(f"FAIL: Content types mismatch: {type(batch[0])}, {type(batch[1])}")
        else:
            print(f"FAIL: Yielded type {type(batch)}")
            
        # Break after first batch
        break

if __name__ == "__main__":
    test_device_data_loader()
