from src.generic_model import GenericModel

from sklearn.ensemble import StackingRegressor
import pandas as pd
import numpy as np

class EnsembleModelRegressor(GenericModel):
    def __init__(self, models, combiner_model=None):
        self.model = StackingRegressor(models, final_estimator=combiner_model)
        self.feature_importances_ = None
        self.fitness_ = 0

    def per_feature_weighted_avg_(self):
        models = self.model.estimators_
        model_scores = np.array([model.fitness_ if model.fitness_ >= 0 else 0 for model in models]).transpose() #1 x M
        model_importances = np.array([model.feature_importances_ for model in models]) #M x F
        weighted_dot = model_scores.dot(model_importances) #1 x F
        net_score = sum(model_scores)
        max_score =  max(model_scores)
        weighted_dot = ((weighted_dot/net_score)*max_score)
        return weighted_dot #1 x F

    def per_feature_weighted_max_(self):
        models = self.model.estimators_
        model_scores = np.array([[model.fitness_ if model.fitness_ >= 0 else 0 for model in models] for i in models]) #M x M
        model_importances = np.array([model.feature_importances_ for model in models]) #M x F
        weighted_dot = model_scores.dot(model_importances) #M x F
        max_vec = np.array([weighted_dot[feature].max() for feature in weighted_dot.transpose()])
        return max_vec #1 x F

    def calculate_importances_(self, method='weighted-avg'):
        methods = {
            "weighted-avg" : self.per_feature_weighted_avg_,
            "weighted-max" : self.per_feature_weighted_max_
        }
        self.feature_importances_ = methods[method]()
