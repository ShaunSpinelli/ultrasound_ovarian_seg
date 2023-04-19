# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun Spinelli 2023/03/01

import logging as lg
# from tqdm import tqdm
# import tqdm.notebook as tq
from tqdm.auto import tqdm

import torch
import torch.nn as nn

# from . import metrics
_logger = lg.getLogger("train")


class Training:
    def __init__(self, metrics, loss, optim, data, epochs, model, save_dir):
        """Training runner

        Args:
            metrics (MetricManager):
            loss (torch.nn.modules.loss):
            optim (torch.optim):
            data (DataLoader):
            epochs (int):
            model ():
            save_dir (str): directory to save model
        """

        self.metrics = metrics
        self.loss = loss
        self.optim = optim
        self.data = data
        self.model = model
        self.epochs = epochs
        self.save_dir = save_dir

        self.step = 0
        self.steps_per_epoch = len(self.data)/self.epochs

    def train_step(self, batch):
        data, labels, fnames = batch
        preds = self.model(data.cuda())
        loss = self.loss(preds, labels.cuda())

        self.metrics.update(preds, labels, self.step)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)
            # if self.step % self.steps_per_epoch == 0:
                # self.metrics.add_image_preds(data, preds, labels, fnames, self.step)
        # _logger.debug(f'Loss: {loss.item()}')

        self.optim.zero_grad()  # zero gradients
        loss.backward()  # calculate gradients
        self.optim.step()  # updated weights

    def save_checkpoint(self):
        """Save checkpoint with current step number"""
        torch.save(self.model.state_dict(), f'{self.save_dir}/model-{self.step}.pth')

    def train_loop(self):
        for i in range(self.epochs):
            # _logger.info(f'Epoch {i}/{self.epochs}')
            print(f'Epoch {i+1}/{self.epochs}')
            for batch in tqdm(self.data):
                self.train_step(batch)
                self.step += 1
            self.metrics.reset()
            self.save_checkpoint()

    def run(self):
        try:
            self.train_loop()
        except KeyboardInterrupt:
            _logger.debug("Quitting due to user cancel")
            self.save_checkpoint()
