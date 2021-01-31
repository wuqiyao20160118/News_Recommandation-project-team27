from train import NRMSModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import hyperParams


model = NRMSModel(hyperParams)
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
    max_epochs=100,
    early_stop_callback=early_stop_config,
    weights_summary="full",
    checkpoint_callback=checkpoints,
    logger=logger
)

trainer.fit(model)
