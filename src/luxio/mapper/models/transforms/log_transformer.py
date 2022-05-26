
from .generic_transformer import GenericTransformer
import numpy as np
import pandas as pd
import pickle


class LogTransformer(GenericTransformer):
    def __init__(self, base=2, add=0, scale=1):
        self.base = base
        self.add = add
        self.scale=scale

    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        return np.nan_to_num(self.transform(X))

    def inverse_transform(self,X):
        return self.base**(X/self.scale) - self.add

    def transform(self,X):
        return np.nan_to_num(self.scale * np.log(X + self.add)/np.log(self.base))
