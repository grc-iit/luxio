
from luxio.mapper.models.common import *
from .generic_metric import GenericMetric
from .scikit_metric import ScikitMetric
from sklearn.metrics import r2_score
import numpy as np
from typing import Union

class r2Metric(ScikitMetric,GenericMetric):
    def __init__(self, mode='avg'):
        self.mode = mode

    def score(self, pred_y:np.array, true_y:np.array, mode:str=None) -> Union[float, np.array]:
        try:
            mode = ScikitMetric.modes[self.mode if mode is None else mode]
            pred_y, true_y = self._input_validator(pred_y, true_y)
            acc = r2_score(pred_y, true_y, multioutput=mode)
            acc = self._nonneg(acc)
            return self._output_validator(pred_y, acc)
        except ValueError:
            return 0
