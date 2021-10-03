
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import copy

class GenericTransformer(ABC,BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.fit_transform(X,y)
        return self

    @abstractmethod
    def fit_transform(self,X,y=None):
        pass

    @abstractmethod
    def inverse_transform(self,X):
        pass

    @abstractmethod
    def transform(self,X):
        pass

    def save(self, path):
        pickle.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pickle.load(open( path, "rb" ))

    def clone(self):
        return copy.deepcopy(self)
