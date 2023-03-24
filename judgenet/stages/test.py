from judgenet.modules import metrics


class Tester():
    def __init__(self, model, test_loader):
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
        return self.metrics
