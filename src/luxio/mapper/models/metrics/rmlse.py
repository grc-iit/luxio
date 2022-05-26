
from luxio.mapper.models.common import *
from .generic_metric import GenericMetric
from .scikit_metric import ScikitMetric
from sklearn.metrics import mean_squared_error as MSE
from typing import Union
import numpy as np

class RMLSEMetric(ScikitMetric,GenericMetric):
    def __init__(self,add=1, mode='avg'):
        self.add = add
        self.mode = mode

    def score(self, pred_y:np.array, true_y:np.array, mode=None) -> Union[float, np.array]:
        try:
            mode = ScikitMetric.modes[self.mode if mode is None else mode]
            pred_y, true_y = self._input_validator(pred_y, true_y)
            rmlse = np.sqrt(MSE(np.log(pred_y + self.add), np.log(true_y + self.add), multioutput=mode))
            return self._output_validator(pred_y, rmlse)
        except ValueError:
            return np.inf
