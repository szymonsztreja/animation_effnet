import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch import Tensor
from torchvision import models


class AnimationEffNet(pl.LightningModule):
    """EfficientNet model for the animation."""

    def __init__(self, lr: float = 0.0001, num_classes: int = 0) -> None:
        """Initialize the FruitVegEffNet model."""
        super().__init__()

        efficientnetB7Model = models.efficientnet_b7(
            weights="EfficientNet_B7_Weights.DEFAULT"
        )

        self.backbone = efficientnetB7Model.features
        self.pooling = efficientnetB7Model.avgpool
        self.dropout = nn.Dropout(inplace=True)
        self.fc1 = nn.LazyLinear(num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )

        self.train_macro_recall = torchmetrics.Recall(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_macro_recall = torchmetrics.Recall(
            num_classes=num_classes, task="multiclass", average="macro"
        )

        self.train_macro_precision = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_macro_precision = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass", average="macro"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pooling(x).flatten(1)

        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc1(x)

        return x

    def training_step(
        self, train_batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Training step of the model."""
        inputs, labels = train_batch

        outputs = self.forward(inputs.float())
        loss = self.loss_function(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        outputs = F.softmax(outputs, dim=1)

        self.train_acc(outputs, labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        self.train_macro_f1(outputs, labels)
        self.log("train_macro_f1", self.train_macro_f1, on_epoch=True, on_step=False)

        self.train_macro_recall(outputs, labels)
        self.log(
            "train_macro_recall", self.train_macro_recall, on_epoch=True, on_step=False
        )

        self.train_macro_precision(outputs, labels)
        self.log(
            "train_macro_precision",
            self.train_macro_precision,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def validation_step(
        self, val_batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Validate step of the model."""
        inputs, labels = val_batch

        outputs = self.forward(inputs.float())
        loss = self.loss_function(outputs, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True)

        outputs = F.softmax(outputs, dim=1)

        self.val_acc(outputs, labels)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

        self.val_macro_f1(outputs, labels)
        self.log("val_macro_f1", self.val_macro_f1, on_epoch=True, on_step=False)

        self.val_macro_recall(outputs, labels)
        self.log(
            "val_macro_recall", self.val_macro_recall, on_epoch=True, on_step=False
        )

        self.val_macro_precision(outputs, labels)
        self.log(
            "val_macro_precision",
            self.val_macro_precision,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer for the model."""
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer