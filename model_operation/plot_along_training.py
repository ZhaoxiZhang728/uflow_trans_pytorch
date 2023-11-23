import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from typing import Any


class LossCurveLogger(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_start(self, trainer, pl_module):
        self.fig, self.ax = plt.subplots()

    def on_train_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        loss = outputs['loss']  # Change this based on how loss is logged in your LightningModule
        self.losses.append(loss.item())

        self.ax.clear()
        self.ax.plot(self.losses, label='Train Loss')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        plt.pause(0.1)  # Update the plot

    def on_train_epoch_end(self, trainer, pl_module):
        plt.show()
