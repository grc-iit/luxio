from .feature_selector import FeatureSelector
import sklearn.ensemble

class RandomForestReducer(FeatureSelector):
    def __init__(self, features, n_features=.5, n_jobs=4, n_estimators=50, max_depth=4):
        super().__init__(features, n_features, n_jobs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.is_fitted_ = False
        self.fitness_ = None

    def _fit_init(self):
        super()._fit_init()
        self.model_ = sklearn.ensemble.RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs, verbose=True)

    def fit_transform(self, X, y=None):
        if not self.is_fitted_:
            self._fit_init()
            self.model_.fit(X, y)
            self.feature_ranks_ = {name:importance for name,importance in zip(self.features, self.model_.feature_importances_)}
            self.sort_features()
            self.is_fitted_ = True
        self.get_support()
        self.fitness_ = self.model_.score(X,y)
        return self.transform(X)