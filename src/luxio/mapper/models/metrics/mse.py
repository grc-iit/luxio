
from luxio.mapper.models.common import *
from .generic_metric import GenericMetric
from .scikit_metric import ScikitMetric
from sklearn.metrics import mean_squared_error as MSE
from typing import Union
import numpy as np

class MSEMetric(ScikitMetric,GenericMetric):
    def __init__(self, mode='avg'):
        self.mode = mode

    def score(self, pred_y:np.array, true_y:np.array, mode:str=None) -> Union[float, np.array]:
        try:
            mode = ScikitMetric.modes[self.mode if mode is None else mode]
            pred_y, true_y = self._input_validator(pred_y, true_y)
            mse = MSE(pred_y, true_y, multioutput=mode)
            return self._output_validator(pred_y, mse)
        except ValueError:
            return np.inf
