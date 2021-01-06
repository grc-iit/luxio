
from sklearn.metrics import r2_score
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin

class GenericModel(ABC,BaseEstimator,RegressorMixin):
    def predict(self, test_x):
        return self.model.predict(test_x)

    def fit_predict(self, train_x, train_y, sample_weight=None):
        self.model.fit(train_x,train_y,sample_weight)
        pred = self.model.predict(train_x)
        self.fitness_ = r2_score(pred,train_y)
        self.calculate_importances_()
        return pred

    def fit(self, train_x, train_y, sample_weight=None):
        self.fit_predict(train_x, train_y, sample_weight)
        return self

    def score(self, test_x, test_y, sample_weight=None):
        return self.model.score(test_x, test_y, sample_weight)

    def rmse(self, test_x, test_y):
        pred = self.predict(test_x)
        return np.sqrt(MSE(pred,test_y))

    @abstractmethod
    def calculate_importances_(self, method='weighted-avg'):
        raise NotImplementedError('call to feature_importances_')
