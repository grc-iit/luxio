
from .generic_transformer import GenericTransformer
import numpy as np
import pandas as pd
import pickle


class ScaleTransformer(GenericTransformer):
    def __init__(self, add=0, scale=1):
        self.add = add
        self.scale=scale

    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        return self.transform(X)

    def inverse_transform(self,X):
        return (X/self.scale) - self.add

    def transform(self,X):
        return self.scale * (X + self.add)
