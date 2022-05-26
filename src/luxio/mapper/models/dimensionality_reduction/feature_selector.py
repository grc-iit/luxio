from .dimension_reducer import DimensionReducer
from luxio.mapper.models.common import *
import numpy as np

class FeatureSelector(DimensionReducer):
    def sort_features(self):
        self.sorted_features_ = list(self.feature_ranks_.items())
        self.sorted_features_.sort(key=lambda x: x[1], reverse=True)
        self.sorted_features_names_ = [name for name, importance in self.sorted_features_]
        return self.sorted_features_names_

    def get_support(self):
        self.selected_features_ = {feature: True for feature in self.sorted_features_names_[0:self.n_features_]}
        self.support_ = np.array([(feature in self.selected_features_) for feature in self.features])
        return self.support_

    def transform(self, X):
        X = make_mat(X)
        return X[:,self.support_]

    def get_selected_feature_names(self):
        return np.array(self.features)[self.support_]