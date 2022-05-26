from luxio.mapper.models.common import *
from .rfe_regressor import *
import sklearn
import pandas as pd
import numpy as np
from copy import copy

from typing import List,Dict,Tuple,Any

class UniOutputModel(RFERegressor):
    def fit(self, X:np.array, y:np.array, sample_weight=None):
        self.fit_predict(X, y, sample_weight)
        return self

    def fit_predict(self, X:np.array, y:np.array, sample_weight=None):
        self._fit_init()
        X = make_mat(X)
        y = make_mat(y)
        #Fit to training data
        self.models_ = [sklearn.base.clone(self.base_model_) for y_col in y.T]
        if hasattr(self.model_, "fit_predict"):
            pred_y = np.array([model.fit_predict(X, y_col) for model,y_col in zip(self.models_,y.T)]).T
        else:
            pred_y = np.array([model.fit(X, y_col).predict(X) for model,y_col in zip(self.models_,y.T)]).T
        assert_shape(pred_y, y)
        #Get fitness scores
        self._fitness(y, pred_y)
        self._feature_importances(X)
        return pred_y

    def predict(self, X):
        X = make_mat(X)
        return np.array([model.predict(X) for model in self.models_]).T