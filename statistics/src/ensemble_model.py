from generic_model import GenericModel

from sklearn.ensemble import StackingRegressor
import pandas as pd
import numpy as np

class EnsembleModelRegressor(GenericModel):
    def __init__(self, models, combiner_model=None):
        self.models_ = dict(models)
        self.model_ = StackingRegressor(models, final_estimator=combiner_model)
        self.features_ = None
        self.vars_ = None
        self.feature_importances_ = None
        self.fitness = 0

    def per_feature_weighted_avg_(self):
        model_scores = np.array([model.fitness_ if model.fitness_ >= 0 else 0 for model in self.models_.values()]).transpose() #1 x M
        model_importances = np.array([model.feature_importances_ for model in self.models_.values()]) #M x F
        weighted_dot = model_scores.dot(model_importances) #1 x F
        net_score = sum(model_scores)
        max_score =  max(model_scores)
        weighted_dot = ((weighted_dot/net_score)*max_score)
        return weighted_dot #1 x F

    def per_feature_weighted_max_(self):
        model_scores = np.array([[model.fitness_ if model.fitness_ >= 0 else 0 for model in self.models_.values()] for i in range(len(self.models_.values()))]) #M x M
        model_importances = np.array([model.feature_importances_ for model in self.models_.values()]) #M x F
        weighted_dot = model_scores.dot(model_importances) #M x F
        max_vec = np.array([weighted_dot[feature].max() for feature in weighted_dot.transpose()])
        return max_vec #1 x F

    def feature_importances_(self, method='weighted-avg'):
        methods = {
            "weighted-avg" : self.per_feature_weighted_avg_,
            "weighted-max" : self.per_feature_weighted_max_
        }
        self.feature_importances_ = methods[method]()
