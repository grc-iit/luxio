from .dimension_reducer import DimensionReducer
from .random_forest import RandomForestReducer
from .variance import VarianceReducer
from .pca import PCAReducer

class DimensionReducerFactory(DimensionReducer):
    methods = {
        "random-forest": "random_forest_",
        "variance": "variance_",
        "pca": "pca_"
    }

    def __init__(self, features=None, n_features=.5, method=None, n_jobs=4):
        super().__init__(features, n_features, n_jobs)
        self.method = method
        self.random_forest_ = None
        self.variance_ = None
        self.is_fitted_ = False

    def set_method(self, method):
        self.method = method

    def _get_method(self):
        return self.__dict__[DimensionReducerFactory.methods[self.method]]

    def fit_transform(self, X, y=None):
        if not self.is_fitted_:
            self._fit_init()
            self.random_forest_ = RandomForestReducer(self.features, self.n_features, n_jobs=self.n_jobs).fit(X,y)
            self.variance_ = VarianceReducer(self.features, self.n_features, n_jobs=self.n_jobs).fit(X,y)
            self.pca_ = PCAReducer(self.features, self.n_features, n_jobs=self.n_jobs).fit(X,y)
            self.is_fitted_ = True
        return self.transform(X)

    def transform(self, X):
        if self.method is None:
            return X
        return self._get_method().transform(X)

    def get_selected_feature_names(self):
        if self.method is None:
            return None
        return self._get_method().get_selected_feature_names()