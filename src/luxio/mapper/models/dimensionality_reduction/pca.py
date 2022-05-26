from .dimension_reducer import DimensionReducer
from sklearn.decomposition import PCA

class PCAReducer(DimensionReducer):
    def __init__(self, features, n_features=.5, n_jobs=4):
        super().__init__(features, n_features, n_jobs)
        self.is_fitted_ = False

    def _fit_init(self):
        super()._fit_init()
        self.model_ = PCA()

    def fit_transform(self, X, y=None):
        if not self.is_fitted_:
            self._fit_init()
            self.model_.fit(X, y)
            self.is_fitted_ = True
        self.model_ = self.model_.set_params(n_components=self.n_features)
        return self.transform(X)

    def transform(self, X):
        return self.model_.transform(X)