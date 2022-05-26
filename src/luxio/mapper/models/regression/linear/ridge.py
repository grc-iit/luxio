from luxio.mapper.models.regression.coeff_importances import RFECoeffImportances
from luxio.mapper.models.regression.multioutput_model import MultiOutputModel
import sklearn.linear_model
from sklearn.feature_selection import RFE
from ray import tune

class RFERidgeRegression(RFECoeffImportances,MultiOutputModel):
    def __init__(self, features=None, n_features=None, feature_thresh=.01, fitness_metric=None, error_metric=None, alpha=.1, fit_intercept=True, solver="auto"):
        super().__init__(features, n_features, fitness_metric, error_metric)
        self.model_ = RFE(
            sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver),
            n_features_to_select=n_features)

    def get_search_space(self, level=0):
        return {
            "alpha" : tune.uniform(.1, 1),
            "fit_intercept" : tune.choice([True, False]),
            "solver" : tune.choice(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        }
