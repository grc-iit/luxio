from luxio.mapper.models.regression.feature_importances import RFEFeatureImportances
from luxio.mapper.models.regression.unioutput_model import UniOutputModel
import xgboost
from sklearn.feature_selection import RFE
from ray import tune

class RFEXGBRegressor(RFEFeatureImportances,UniOutputModel):
    def __init__(self, features=None, n_features=None, fitness_metric=None, error_metric=None, n_estimators=3, eta=0, max_depth=6):
        super().__init__(features, n_features, fitness_metric, error_metric)
        self.base_model_ = self.model_ = RFE(
            xgboost.XGBRegressor(n_estimators=n_estimators, eta=eta, max_depth=max_depth),
            n_features_to_select=n_features)

    def get_search_space(self, level=0):
        if level == 0:
            return {
                "n_estimators": tune.randint(3,20),
                "eta": tune.uniform(0,1),
                "max_depth": tune.randint(3,6)
            }
        if level == 1:
            return {
                "n_estimators": tune.randint(3,200),
                "eta": tune.uniform(0,1),
                "max_depth": tune.randint(3,10)
            }
        raise Exception("Level {} not found in XGB".format(level))
