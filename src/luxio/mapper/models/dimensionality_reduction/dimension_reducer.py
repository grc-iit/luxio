
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import copy

class DimensionReducer(ABC,BaseEstimator,TransformerMixin):
    def __init__(self, features, n_features, n_jobs, random_seed=132415):
        self.features = features
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.random_seed = random_seed

    def set_n_features(self, n_features):
        self.n_features = n_features

    def get_selected_feature_names(self):
        """
        Some dimension reduction techniques remove feature names.
        For those that maintain them, this function will be overriden.
        """
        return [str(i) for i in range(self.n_features_)]

    def _fit_init(self):
        if isinstance(self.n_features, int):
            self.n_features_ = self.n_features
        else:
            self.n_features_ = int(len(self.features)*self.n_features)
            if self.n_features_ == 0:
                self.n_features_ = 1

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    @abstractmethod
    def fit_transform(self,X,y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def inverse_transform(self,X):
        return None

    def save(self, path):
        pickle.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pickle.load(open( path, "rb" ))

    def clone(self):
        return copy.deepcopy(self)
