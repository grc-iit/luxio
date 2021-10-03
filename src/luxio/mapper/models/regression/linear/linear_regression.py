from luxio.mapper.models.regression.curve_mixin import RFECoeffImportances
from luxio.mapper.models.regression.multioutput_model import MultiOutputModel
import sklearn.linear_model
from sklearn.feature_selection import RFE
from ray import tune

class RFELinearRegression(RFECoeffImportances,MultiOutputModel):
    def __init__(self, features=None, n_features=None, fitness_metric=None, error_metric=None, fit_intercept=False):
        super().__init__(features, n_features, fitness_metric, error_metric)
        self.model_ = RFE(
            sklearn.linear_model.LinearRegression(fit_intercept=fit_intercept),
            n_features_to_select=n_features)

    def get_search_space(self, level=0):
        return {"fit_intercept" : tune.choice([True, False])}
