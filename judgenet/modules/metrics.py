import torch


def MSE(pred_mu, label):
    return ((pred_mu - label) ** 2).mean()


def entropy(pred_dist):
    return (pred_dist.entropy()).mean()


def log_prob(pred_dist, label):
    return (pred_dist.log_prob(label)).mean()


class Accuracy():
    '''Accuracy metric. ratio of correct predictions to total predictions
    Attributes:
        correct [int]: number of correct predictions
        total [int]: number of total predictions
    '''

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.correct += torch.sum(preds == labels).item()
        self.total += len(labels)

    def finalize(self) -> float:
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: accuracy of the model
        '''
        if self.total == 0:
            return 1.0
        return self.correct / self.total


class BinaryPrecision():
    '''Binary Precision metric. ratio of true positive predictions to total
    positive predictions.
    Attributes:
        true_positive [int]: number of true positive predictions
        predicted_positive [int]: number of predicted positive predictions
    '''

    def __init__(self):
        self.true_positive = 0
        self.predicted_positive = 0

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        true_idxs = torch.where(labels == 1)
        self.true_positive += torch.sum(preds[true_idxs]).item()
        self.predicted_positive += torch.sum(preds).item()

    def finalize(self) -> float:
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary precision of the model
        '''
        if self.predicted_positive == 0:
            return 1.0
        return self.true_positive / self.predicted_positive
    

class Precision():
    '''Precision metric. ratio of true positive predictions to total
    positive predictions.
    Attributes:
        true_positive [int]: number of true positive predictions
        predicted_positive [int]: number of predicted positive predictions
    '''

    def __init__(self, n_cats):
        self.n_cats = n_cats
        self.true_positive = {cat: 0 for cat in range(n_cats)}
        self.predicted_positive = {cat: 0 for cat in range(n_cats)}

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        for curlabel in range(self.n_cats):
            true_idxs = torch.where(labels == curlabel)
            self.true_positive[curlabel] += torch.sum(preds[true_idxs]).item()
            self.predicted_positive[curlabel] += torch.sum(preds).item()

    def finalize(self) -> float:
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary precision of the model
        '''
        if self.predicted_positive == 0:
            return 1.0
        return self.true_positive / self.predicted_positive


class BinaryRecall():
    '''Binary Recall metric. ratio of true positive predictions to total
    positive labels.
    Attributes:
        true_positive [int]: number of true positive predictions
        gt_positive [int]: number of positive labels
    '''

    def __init__(self):
        self.true_positive = 0
        self.gt_positive = 0

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        true_idxs = torch.where(labels == 1)
        self.true_positive += torch.sum(preds[true_idxs]).item()
        self.gt_positive += torch.sum(labels).item()

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        if self.gt_positive == 0:
            return 1.0
        return self.true_positive / self.gt_positive


class BinaryF1():
    '''Binary F1 metric.
    '''

    def __init__(self):
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        final_precision = self.precision.finalize()
        final_recall = self.recall.finalize()
        if final_precision + final_recall == 0:
            return 1.0
        return (2 * final_precision * final_recall) / (final_precision + final_recall)
