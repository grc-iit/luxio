
from .generic_transformer import GenericTransformer
import numpy as np
import pickle


class ChainTransformer(GenericTransformer):

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self,X,y=None):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
        return self

    def fit_transform(self,X,y=None):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
        return X

    def inverse_transform(self,X):
        for transformer in reversed(self.transformers):
            X = transformer.inverse_transform(X)
        return X

    def transform(self,X):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X
