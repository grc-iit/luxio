
from abc import ABC, abstractmethod
import pickle
import numpy as np
from typing import Tuple, Union
import numbers

class GenericMetric(ABC):
    def __init__(self):
        return

    @abstractmethod
    def score(self, pred_y:np.array, true_y:np.array, mode=None) -> np.array:
        return

    def _input_validator(self, pred_y:np.array, true_y:np.array) -> Tuple[np.array, np.array]:
        pred_y = np.array(pred_y)
        true_y = np.array(true_y)
        if pred_y.shape != true_y.shape:
            raise Exception(f"Input mismatch to score: pred_dims = {pred_y.shape} true_dims = {true_y.shape}")
        if pred_y.ndim > 2 or true_y.ndim > 2:
            raise Exception(f"Scoring functions work on only 1-D or 2-D data: {pred_y.shape} {true_y.shape}")
        return pred_y, true_y

    @staticmethod
    def _nonneg(scores:Union[float,np.array]) -> Union[float,np.array]:
        if isinstance(scores, numbers.Number):
            return scores if scores > 0 else 0
        scores[scores < 0] = 0
        return scores
