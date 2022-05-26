from .generic_regressor import GenericRegressor
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import pprint, warnings

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

pp = pprint.PrettyPrinter(depth=6)

class RFERegressor(GenericRegressor):
    def __init__(self, features=None,
                 transform=None, transform_method=None, transform_y=None,
                 heuristic_reducer=None, reducer_method=None,
                 n_features_heur=None, n_features=None,
                 fitness_metric=None, error_metric=None):
        super().__init__(features, transform, transform_method, transform_y, heuristic_reducer, reducer_method, n_features_heur, n_features, fitness_metric, error_metric)

    def _make_pipeline(self, model):
        pipeline = []
        if self.transform:
            self.transform_ = self.transform.clone()
            self.transform_.set_method(self.transform_method)
            pipeline.append(("transform", self.transform_))
        if self.heuristic_reducer:
            self.heuristic_reducer_ = self.heuristic_reducer.clone()
            self.heuristic_reducer_.set_method(self.reducer_method)
            pipeline.append(("heuristic_reducer", self.heuristic_reducer_))
        if self.n_features is not None:
            pipeline.append(("feature_eliminator", RFE(model, n_features_to_select=self.n_features)))
        else:
            pipeline.append(("feature_eliminator", RFE(model, n_features_to_select=1.0)))
        return Pipeline(pipeline)

    @abstractmethod
    def _per_model_importances(self, model):
        return

    def _expand_to_support(self, importances, support):
        j = 0
        for i in range(len(support)):
            support[i] = importances[j] if support[i] else 0
            j += 1
        return support

    def _feature_importances(self, X):
        if self.heuristic_reducer is not None:
            heuristic_reducer = self.model_.named_steps["heuristic_reducer"]
            self.features_ = heuristic_reducer.get_selected_feature_names()
        else:
            self.features_ = self.features

        if self.model_ is not None:
            rfe_model = self.model_.named_steps["feature_eliminator"]
            self.features_ = list(np.array(self.features_)[rfe_model.support_])
            self.feature_importances_ = self._per_model_importances(X, rfe_model.estimator_)
        if self.models_ is not None:
            importances_matrix = np.array([
                self._expand_to_support(self._per_model_importances(X, rfe_model.estimator_), rfe_model.support_)
                for rfe_model in self.models_])
            self.feature_importances_ = np.average(importances_matrix * self.scores_, axis=0)

    def named_importances(self, feature_names:List[str]=None, order=False) -> List[Tuple[str,float]]:
        if feature_names is None:
            feature_names = self.features_
        importances = [(feature_name, importance) for feature_name,importance in zip(feature_names, self.feature_importances_)]
        if order == True:
            importances.sort(key=lambda x : x[1], reverse=True)
        return importances

    def save_importances(self, path, order=False, features=None) -> None:
        if features is None:
            features = self.features_
        df = pd.DataFrame(self.named_importances(features, order=order), columns=["feature","importance"])
        df.to_csv(path, index=False)