
from .generic_metric import GenericMetric
from sklearn.metrics import mean_squared_error as MSE
from .scikit_metric import ScikitMetric
import numpy as np

class RMSEMetric(ScikitMetric,GenericMetric):
    def __init__(self, mode='avg'):
        self.mode = mode

    def score(self, pred_y, true_y, mode:str=None):
        try:
            mode = ScikitMetric.modes[self.mode if mode is None else mode]
            pred_y, true_y = self._input_validator(pred_y, true_y)
            rmse = np.sqrt(MSE(pred_y, true_y, multioutput=mode))
            return self._output_validator(pred_y, rmse)
        except ValueError:
            return np.inf
