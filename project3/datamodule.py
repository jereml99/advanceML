import lightning as L
import torch
from torch.utils.data import random_split
import torch.utils.data as data
from torch_geometric.datasets import TUDataset


class MyDataModule(L.LightningDataModule):
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
        return data.DataLoader(self.train_dataset)

    def val_dataloader(self):
        return data.DataLoader(self.validation_dataset)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset)
