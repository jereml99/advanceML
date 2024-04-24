import lightning as L
import torch
from torch.utils.data import random_split
import torch.utils.data as data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

FEATURE_DIM = 7

class TUDataMoudle(L.LightningDataModule):
    def prepare_data(self):
        TUDataset(root="./data/", name="MUTAG")

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = TUDataset(root="./data/", name="MUTAG")
        rng = torch.Generator().manual_seed(0)
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
            dataset, (100, 44, 44), generator=rng
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=100)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=44)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
