from lightning.pytorch.cli import LightningCLI

from model import VAE
from datamodule import TUDataMoudle

if __name__ == "__main__":
        LightningCLI(
        VAE,
        TUDataMoudle,
    )