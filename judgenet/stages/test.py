import os

import torch

from judgenet.modules import metrics
from judgenet.utils.file import write_json
from judgenet.values.constants import test_metrics_filename


class Tester():
    def __init__(self, exp_name, exp_dir, model, test_loader):
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.model = model.eval()
        self.test_loader = test_loader
        self.metrics = {
            "mse": 0,
            "entropy": 0,
            "log_prob": 0,
        }

    def run(self):
        for features, labels in self.test_loader:
            pred_mu, pred_dist = self.model.predict(features)
            self.metrics["mse"] += metrics.MSE(pred_mu, labels)
            self.metrics["entropy"] += metrics.entropy(pred_dist)
            self.metrics["log_prob"] += metrics.log_prob(pred_dist, labels)
        for metric in self.metrics:
            self.metrics[metric] = self.metrics[metric].item() / len(self.test_loader.dataset)
        write_json(self.metrics, os.path.join(
            self.exp_dir, f"{self.exp_name}_{test_metrics_filename}"))
        return self.metrics


class TesterClassification():
    def __init__(self, exp_name, exp_dir, model, test_loader):
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.model = model.eval()
        self.test_loader = test_loader
        self.metrics = {
            "accuracy": 0,
        }

    def run(self):
        for features, labels in self.test_loader:
            pred = self.model.predict(features)
            self.metrics["accuracy"] += torch.sum(pred == labels)
        for metric in self.metrics:
            self.metrics[metric] = self.metrics[metric].item() / len(self.test_loader.dataset)
        write_json(self.metrics, os.path.join(
            self.exp_dir, f"{self.exp_name}_{test_metrics_filename}"))
        return self.metrics