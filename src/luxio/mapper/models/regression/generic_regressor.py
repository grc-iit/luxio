#Transform -> Heuristic Dimension Reduce -> Feature Select + Fitting

from luxio.mapper.models.common import *
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
import copy
import pprint, warnings

pp = pprint.PrettyPrinter(depth=6)

class GenericRegressor(BaseEstimator,RegressorMixin,ABC):
    def __init__(self, features, transform, transform_method, transform_y, heuristic_reducer, reducer_method, n_features_heur, n_features, fitness_metric, error_metric):
        super().__init__()
        self.features = features
        self.transform = transform
        self.transform_method = transform_method
        self.transform_y = transform_y
        self.heuristic_reducer = heuristic_reducer
        self.reducer_method = reducer_method
        self.n_features_heur = n_features_heur
        self.n_features = n_features
        self.fitness_metric = fitness_metric
        self.error_metric = error_metric

        self.transform_y_ = None
        self.fitness_ = 0
        self.error_ = 0
        self.fitnesses_ = None
        self.errors_ = None
        self.feature_importances_ = None
        self.features_ = None
        self.model_ = None
        self.models_ = None

    @abstractmethod
    def _fit_init(self):
        return

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        return None

    @abstractmethod
    def fit_predict(self, X, y, sample_weight=None):
        return None

    @abstractmethod
    def predict(self, X):
        return None

    def _fitness(self, y, pred_y):
        self.fitness_ = self.fitness_metric.score(pred_y, y)
        self.error_ = self.error_metric.score(pred_y, y)
        self.fitnesses_ = self.fitness_metric.score(pred_y, y, mode='all')
        self.errors_ = self.error_metric.score(pred_y, y, mode='all')

    def score(self, X, y, sample_weight=None):
        pred_y = self.predict(X)
        return self.fitness_metric.score(pred_y,y)

    def error(self, X, y) -> float:
        pred_y = self.predict(X)
        return self.error_metric.score(pred_y, y)

    def analyze(self, X=None, y=None, metrics=None) -> Dict[str,Any]:
        X = make_mat(X)
        y = make_mat(y)
        if metrics == None:
            metrics = {}
        pred_y = self.predict(X)
        return {
            "orig-fitness" : self.fitness_,
            "orig-error" : self.error_,
            "new-fitness" : self.fitness_metric.score(pred_y, y) if pred_y is not None else 0,
            "new-error" : self.error_metric.score(pred_y, y) if pred_y is not None else 0,
            "metrics" : { metric_id : metric.score(pred_y, y) for metric_id,metric in metrics.items() } if pred_y is not None else None
        }

    def clone(self):
        return copy.deepcopy(self)

    def save(self, path) -> None:
        pickle.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pickle.load(open( path, "rb" ))

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def deserialize(bin:bytes):
        return pickle.loads(bin)
