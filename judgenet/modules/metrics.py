import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


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
            self.true_positive[curlabel] += torch.sum(preds[true_idxs] == curlabel).item()
            self.predicted_positive[curlabel] += torch.sum(preds == curlabel).item()

    def finalize(self) -> float:
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary precision of the model
        '''
        precisions = {cat: 0 for cat in range(self.n_cats)}
        for curlabel in range(self.n_cats):
            if self.predicted_positive[curlabel] == 0:
                precisions[curlabel] = 1.0
            else:
                precisions[curlabel] = self.true_positive[curlabel] / \
                    self.predicted_positive[curlabel]
        return precisions


class Recall():
    '''Recall metric. ratio of true positive predictions to total
    positive labels.
    Attributes:
        true_positive [int]: number of true positive predictions
        gt_positive [int]: number of positive labels
    '''

    def __init__(self, n_cats):
        self.n_cats = n_cats
        self.true_positive = {cat: 0 for cat in range(n_cats)}
        self.gt_positive = {cat: 0 for cat in range(n_cats)}

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        for curlabel in range(self.n_cats):
            true_idxs = torch.where(labels == curlabel)
            self.true_positive[curlabel] += torch.sum(
                preds[true_idxs] == curlabel).item()
            self.gt_positive[curlabel] += torch.sum(
                labels == curlabel).item()

    def finalize(self) -> float:
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary precision of the model
        '''
        recalls = {cat: 0 for cat in range(self.n_cats)}
        for curlabel in range(self.n_cats):
            if self.gt_positive[curlabel] == 0:
                recalls[curlabel] = 1.0
            else:
                recalls[curlabel] = self.true_positive[curlabel] / \
                    self.gt_positive[curlabel]
        return recalls
    

class F1():
    '''F1 metric.
    '''

    def __init__(self, n_cats):
        self.n_cats = n_cats
        self.precision = Precision(n_cats)
        self.recall = Recall(n_cats)

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
        f1_sum = 0
        for curlabel in range(self.n_cats):
            if final_precision[curlabel] + final_recall[curlabel] == 0:
                f1_sum += 1.0
            else:
                f1_sum += ((2 * final_precision[curlabel] * final_recall[curlabel]) /
                       (final_precision[curlabel] + final_recall[curlabel]))
        return f1_sum / self.n_cats


# class R_2():
#     '''R Squared metric.
#     '''

#     def __init__(self):
#         self.rss = 0
#         self.sum_of_preds = 0
#         self.sum_of_preds_squared = 0
#         self.sum_of_labels = 0
#         self.n_total = 0

#     def update(self, preds, labels):
#         '''Update the metric given a new batch of prediction-label pairs.
#         '''
#         assert len(preds) == len(labels)
#         self.rss += torch.sum(torch.square(preds - labels)).item()
#         self.sum_of_preds += torch.sum(preds).item()
#         self.sum_of_preds_squared += torch.sum(torch.square(preds)).item()
#         self.sum_of_labels += torch.sum(labels).item()
#         self.n_total += len(labels)

#     def finalize(self):
#         '''Finalizes the accuracy computation and returns the metric value.
#         Returns:
#             value [float]: binary recall of the model
#         '''
#         label_mean = self.sum_of_labels / self.n_total
#         tss = self.sum_of_preds_squared - (
#             2 * label_mean * self.sum_of_preds) + (self.n_total * label_mean * label_mean)
#         return 1 - (self.rss / tss)


class R_2():
    '''R Squared metric.
    '''

    def __init__(self):
        self.rss = RSS()
        self.tss = TSS()

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.rss.update(preds, labels)
        self.tss.update(preds, labels)

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        return 1 - (self.rss.finalize() / self.tss.finalize())


class RSS():
    '''RSS metric.
    '''

    def __init__(self):
        self.rss = 0
        self.n_total = 0

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.rss += F.mse_loss(preds, labels, reduction="sum").item()
        self.n_total += len(labels)

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        return self.rss / self.n_total


class TSS():
    '''
    '''

    def __init__(self):
        self.tss = 0
        # self.hardcode_mean = 5.0  # RecommendHiring
        # self.hardcode_mean = 5.0  # Engaging
        # self.hardcode_mean = 5.372117  # Smiled
        # self.hardcode_mean = (5.0 - 2.030430) / \
        #     (7.000000 - 3.000000)  # RecommendHiring
        # self.hardcode_mean = (5.0 - 2.030430) / \
        #     (6.745900 - 2.144769)  # Engaging
        # self.hardcode_mean = (5.372117 - 2.030430) / \
        #     (7.000000 - 1.500000)  # Smiled
        self.hardcode_mean = 0
        self.n_total = 0

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.tss += F.mse_loss(torch.full_like(labels, self.hardcode_mean), labels, reduction="sum").item()
        self.n_total += len(labels)

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        return self.tss / self.n_total


class AUC():
    '''
    '''

    def __init__(self):
        self.preds = []
        self.labels = []

    def update(self, preds, labels):
        '''Update the metric given a new batch of prediction-label pairs.
        '''
        assert len(preds) == len(labels)
        self.preds.append(preds)
        self.labels.append(labels)

    def finalize(self):
        '''Finalizes the accuracy computation and returns the metric value.
        Returns:
            value [float]: binary recall of the model
        '''
        return roc_auc_score(torch.cat(self.labels), torch.cat(self.preds))
