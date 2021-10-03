from luxio.mapper.models.regression.feature_importances import RFEFeatureImportances
from luxio.mapper.models.regression.multioutput_model import MultiOutputModel
import sklearn.ensemble
import numpy as np
from ray import tune

class RFERandomForestRegressor(RFEFeatureImportances,MultiOutputModel):
    def __init__(self, features=None,
                 transform=None, transform_method=None, transform_y=None,
                 heuristic_reducer=None, reducer_method=None,
                 n_features_heur=None, n_features=None,
                 fitness_metric=None, error_metric=None,
                 n_estimators=3, max_depth=6, max_leaf_nodes=None):
        super().__init__(features, transform, transform_method, transform_y, heuristic_reducer, reducer_method, n_features_heur, n_features, fitness_metric, error_metric)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

    def _fit_init(self):
        self.model_ = self._make_pipeline(sklearn.ensemble.RandomForestRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            max_leaf_nodes = self.max_leaf_nodes
        ))

    def get_search_space(self, level=0):
        if level == 0:
            return {
                "n_estimators": tune.randint(3,20),
                "max_depth": tune.randint(3,6),
                "max_leaf_nodes": tune.randint(2**3, 2**12)
            }
        if level == 1:
            return {
                "n_estimators": tune.randint(3,200),
                "max_depth": tune.randint(3,20),
                "max_leaf_nodes": tune.randint(8, 2**20)
            }
        raise Exception("Level {} not found in RF".format(level))
