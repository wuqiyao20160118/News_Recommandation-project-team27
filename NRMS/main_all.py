from train_deep_cross import NRMSCrossModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import hyperParams
import os


def train():
    model = NRMSCrossModel(hyperParams)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    checkpoints = ModelCheckpoint(
        filepath=f'lightning_logs/{hyperParams["description"]}/{hyperParams["version"]}/' + '{epoch}-{auroc:.2f}',
        save_top_k=3,
        verbose=True,
        monitor="auroc",
        mode="max",
        save_last=True
    )
    early_stop_config = EarlyStopping(
        monitor="auroc",
        min_delta=0.01,
        patience=5,
        strict=False,
        verbose=True,
        mode="max"
    )
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=hyperParams["description"],
        version=hyperParams["version"]
    )
    trainer = Trainer(
        max_epochs=300,
        gpus=1,
        callbacks=[early_stop_config, checkpoints],
        weights_summary="full",
        logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
