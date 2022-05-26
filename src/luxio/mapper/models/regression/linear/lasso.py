from luxio.mapper.models.regression.curve_mixin import RFECoeffImportances
from luxio.mapper.models.regression.multioutput_model import MultiOutputModel
import sklearn.linear_model
from sklearn.feature_selection import RFE
from ray import tune

class RFELassoRegression(RFECoeffImportances,MultiOutputModel):
    def __init__(self, features=None, n_features=None, fitness_metric=None, error_metric=None, alpha=.1, fit_intercept=True, selection="random"):
        super().__init__(features, n_features, fitness_metric, error_metric)
        self.model_ = RFE(
            sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, selection=selection),
            n_features_to_select=n_features)

    def get_search_space(self, level=0):
        return {
            "alpha" : tune.uniform(.1, 1),
            "fit_intercept" : tune.choice([True, False]),
            "selection" : tune.choice(["random", "cyclic"])
        }
