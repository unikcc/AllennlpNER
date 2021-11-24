from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support, get_entities

class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, index2token):
        self.index2token = index2token
        self.reset()
        # self.predicts, self.golds = [], []
        self.tp, self.fn, self.fp = 0, 0, 0

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions = predictions.tolist()
        gold_labels = gold_labels.tolist()
        mask = mask.tolist()
        for i in range(len(predictions)):
            curr_length = sum(mask[i])
            predict = predictions[i][:curr_length]
            gold = gold_labels[i][:curr_length]

            predict = [self.index2token[w] for w in predict]
            gold = [self.index2token[w] for w in gold]

            predict_entities = get_entities(predict)
            gold_entities = get_entities(gold)

            predict_entities = set(predict_entities)
            gold_entities = set(gold_entities)

            true_positive = predict_entities.intersection(gold_entities)

            self.tp += len(true_positive)
            self.fp += len(predict_entities)
            self.fn += len(gold_entities)

    @overrides
    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        # precision = float(self._true_positives) / (float(self._true_positives + self._false_positives) + 1e-13)
        # recall = float(self._true_positives) / (float(self._true_positives + self._false_negatives) + 1e-13)
        # f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        # print(self.predicts)
        # print(self.golds)
        # p, r, f, true_sum = precision_recall_fscore_support(self.golds, self.predicts, zero_division=0)
        p = self.tp / self.fp if self.tp * self.fp > 0 else 0
        r = self.tp / self.fn if self.tp * self.fn > 0 else 0
        f = 2 * p * r / (p + r) if p * r > 0 else 0

        # Reset counts if at end of epoch.
        if reset:
            self.reset()
        res = {
            'precision': p,
            'recall': r,
            'f1_score': f
        }
        return res

        return p, r, f, true_sum

    @overrides
    def reset(self):
        self.predicts = []
        self.golds = []
