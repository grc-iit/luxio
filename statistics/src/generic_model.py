
from sklearn.metrics import r2_score
import numpy as np
from abc import ABC, abstractmethod

class GenericModel(ABC):
    def predict(self, test_x):
        return self.ensemble_.predict(test_x)

    def fit_predict(self, train_x, train_y):
        pred = self.model_.fit_predict(test_x,test_y)
        self.fitness = r2_score(pred,train_y)
        self.feature_importances_()
        return pred

    def fit(self, train_x, train_y):
        self.fit_predict(train_x, train_y)

    def score(self, test_x, test_y):
        return self.model_.score(test_x, test_y)

    def rmse(self, test_x, test_y):
        pred = self.predict(test_x)
        return np.sqrt(MSE(pred,test_y))
    
    @abstractmethod
    def feature_importances_(self):
        pass
