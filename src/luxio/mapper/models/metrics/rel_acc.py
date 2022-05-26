
from luxio.mapper.models.common import *
from .generic_metric import GenericMetric
from .custom_metric import CustomMetric
from sklearn.metrics import mean_squared_error as MSE
from typing import Union
import numpy as np

class RelativeAccuracyMetric(CustomMetric,GenericMetric):
    def __init__(self, add=1, scale=1, mode='avg'):
        self.add = add
        if mode == 'avg':
            self.row_mode = 'avg'
            self.col_mode = 'avg'
        elif mode == 'all':
            self.row_mode = 'avg'
            self.col_mode = 'all'
        self.scale = scale

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
            rel_acc = (self.scale - rel_err)/self.scale
            if rel_acc.ndim == 2:
                rel_acc = row_mode(rel_acc, axis=0)
                rel_acc = col_mode(rel_acc, axis=rel_acc.ndim - 1)
            else:
                rel_acc = row_mode(rel_acc)
            return self._nonneg(rel_acc)
        except ValueError:
            return 0
