from .feature_selector import FeatureSelector
import numpy as np

class VarianceReducer(FeatureSelector):
    def __init__(self, features, n_features=.5, n_jobs=4):
        super().__init__(features, n_features, n_jobs)
        self.is_fitted_ = False

    def fit_transform(self, X, y=None):
        if not self.is_fitted_:
            self._fit_init()
            variances = np.var(X, axis=0)
            self.feature_ranks_ = {name:importance for name,importance in zip(self.features, variances)}
            self.sort_features()
            self.is_fitted_ = True
        self.get_support()
        return self.transform(X)