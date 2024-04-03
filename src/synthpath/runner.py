import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

def main():
    cli = LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()