import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import AnimationTypesDatamodule
from EfficientNet import AnimationEffNet


if __name__ == "__main__":
    dm = AnimationTypesDatamodule()
    dm.setup()
    num_classes = len(dm.train_dataset.dataset.classes)
    print(num_classes)
    print(torch.cuda.is_available())

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/fruit_veg_effnet",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )
    model = AnimationEffNet(num_classes=num_classes)
    logger = TensorBoardLogger("lightning_logs", name="effnet")
    trainer = pl.Trainer(
        accelerator="cuda",
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=30,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)