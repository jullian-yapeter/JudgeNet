import os

from judgenet.modules.metrics import F1, R_2, Accuracy
from judgenet.utils.file import write_json
from judgenet.values.constants import test_metrics_filename


class Tester():
    def __init__(self, exp_name, exp_dir, model, test_loader):
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.model = model.eval()
        self.test_loader = test_loader
        self.metrics = None

    def run(self):
        for features, labels in self.test_loader:
            preds = self.model.predict(features)
            for metric_name in self.metrics:
                self.metrics[metric_name].update(preds, labels)
        results = {metric_name: metric.finalize() for metric_name, metric in self.metrics.items()}
        write_json(results, os.path.join(
            self.exp_dir, f"{self.exp_name}_{test_metrics_filename}"))
        return results


class TesterClassification(Tester):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "Accuracy": Accuracy(),
            "F1": F1()
        }


class TesterRegression(Tester):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "R_2": R_2(),
        }
