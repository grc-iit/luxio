from luxio.mapper.models.common import *
from .rfe_regressor import *
import sklearn
import pandas as pd
import numpy as np
from copy import copy

from typing import List,Dict,Tuple,Any

class MultiOutputModel(RFERegressor):
    def fit(self, X:np.array, y:np.array, sample_weight=None):
        self.fit_predict(X,y, sample_weight)
        return self

    def fit_predict(self, X:np.array, y:np.array, sample_weight=None):
        self._fit_init()
        X = make_mat(X)
        y = make_mat(y)

        #Transform predictor data
        if self.transform_y:
            self.transform_y_ = self.transform_y.clone()
            y_trans = self.transform_y_.transform(y)
        else:
            y_trans = y

        #Fit to training data
        if hasattr(self.model_, "fit_predict"):
            pred_y_trans = self.model_.fit_predict(X, y_trans)
        else:
            pred_y_trans = self.model_.fit(X, y_trans).predict(X)

        #Inverse transform predicions
        if self.transform_y:
            pred_y = self.transform_y_.inverse_transform(pred_y_trans)
        else:
            pred_y = pred_y_trans

        #Get fitness scores
        assert_shape(pred_y, y)
        self._fitness(y, pred_y)
        self._feature_importances(X)
        return pred_y

    def predict(self, X):
        X = make_mat(X)
        if self.transform_y:
            return self.transform_y_.inverse_transform(self.model_.predict(X))
        else:
            return self.model_.predict(X)