import torch
import os

from crl_kino.estimator.network import EstimatorModel, ClassifierModel


def load_estimator(estimator_path, classifier_path):
    estimator = EstimatorModel()
    estimator.load_state_dict(torch.load(estimator_path, map_location=torch.device('cpu')))

    classifier = ClassifierModel()
    classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))

    return estimator, classifier
