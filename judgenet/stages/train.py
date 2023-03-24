import copy
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from judgenet.values.constants import final_model_filename


class Trainer():
    def __init__(self,
                 exp_dir,
                 model,
                 train_loader,
                 val_loader,
                 epochs,
                 lr
                 ):
        self.exp_dir=exp_dir
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999))

    def run(self):
        for epoch in tqdm(range(int(self.epochs))):
            ep_losses = []
            self.model.train()
            for features, labels in self.train_loader:
                self.optimizer.zero_grad()
                pred_dist = self.model(features)
                loss = self.model.loss(pred_dist, labels)
                ep_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for features, labels in self.val_loader:
                    pred_dist = self.model(features)
                    loss = self.model.loss(pred_dist, labels)
                    val_losses.append(loss.item())
            if epoch % (int(self.epochs) / 10) == 0:
                print(
                    f"ep:{epoch + 1}, loss:{np.mean(ep_losses)}, val_loss:{np.mean(val_losses)}")
        torch.save(self.model.state_dict(), os.path.join(
            self.exp_dir, final_model_filename))
        return copy.deepcopy(self.model)
