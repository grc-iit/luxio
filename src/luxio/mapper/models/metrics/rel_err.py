
from luxio.mapper.models.common import *
from .generic_metric import GenericMetric
from .custom_metric import CustomMetric
from sklearn.metrics import mean_squared_error as MSE
from typing import Union
import numpy as np

class RelativeErrorMetric(CustomMetric,GenericMetric):
    def __init__(self, add=1, mode='avg'):
        self.add = add
        if mode == 'avg':
            self.row_mode = 'avg'
            self.col_mode = 'avg'
        elif mode == 'all':
            self.row_mode = 'avg'
            self.col_mode = 'all'

    def score(self, pred_y:np.array, true_y:np.array, mode=None) -> Union[float, np.array]:
        if mode == 'all':
            row_mode = 'avg'
            col_mode = 'all'
        try:
            row_mode = CustomMetric.modes[self.row_mode if mode is None else row_mode]
            col_mode = CustomMetric.modes[self.col_mode if mode is None else col_mode]
            pred_y, true_y = self._input_validator(pred_y, true_y)
            diff = np.absolute(true_y - pred_y)
            act = np.absolute(true_y) + self.add
            rel_err = diff/act
            if rel_err.ndim == 2:
                rel_err = row_mode(rel_err, axis=0)
                rel_err = col_mode(rel_err, axis=rel_err.ndim - 1)
            else:
                rel_err = row_mode(rel_err)
            return rel_err
        except ValueError:
            return np.inf
