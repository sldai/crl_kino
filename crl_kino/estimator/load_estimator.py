import torch
import os

from crl_kino.estimator.network import EstimatorModel


def load_estimator(model_path):
    estimator = EstimatorModel(1)
    estimator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return estimator
