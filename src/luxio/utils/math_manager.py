from luxio.common.error_codes import *
import pandas as pd
import numpy as np
from typing import Tuple, List
from clever.transformers import *

class MathManager:
    def __init__(self):
        pass

    def _transform(self, df:pd.DataFrame, features:List[str], transform:GenericTransformer) -> pd.DataFrame:
        return transform.transform(df[features])

    def _score_weight(self, features:List[str], weights:pd.DataFrame) -> float:
        return weights[features].to_numpy().sum()

    def _avg(self, score_id:str, score_program:dict, df:pd.DataFrame, weights:pd.DataFrame, transform:GenericTransformer) -> Tuple[float,float]:
        features = weights.columns.intersection(score_program["features"])

        return

    def _frac(self, score_id:str, score_program:dict, df:pd.DataFrame, weights:pd.DataFrame, transform:GenericTransformer) -> Tuple[float,float]:
        return

    def score(self, score_conf:dict, df:pd.DataFrame, weights:pd.DataFrame=None, transform:GenericTransformer=None) -> Tuple[List[str], np.array]:
        scores = list(score_conf.keys())
        score_weights = []
        for score_id,score_program in score_conf.items():
            if score_program["agg"] == "avg":
                score_weight,score = self._avg(score_id, score_program, df, None, transform)
            if score_program["agg"] == "weighted-avg":
                score_weight,score = self._avg(score_id, score_program, df, weights, transform)
            if score_program["agg"] == "frac":
                score_weight,score = self._frac(score_id, score_program, df, weights, transform)
            score_weights.append(score_weight)
            df.loc[:,score_id] = score
        return scores,np.array(score_weights)

    def transform(self, score_conf:dict, df:pd.DataFrame, weights:pd.DataFrame=None, transform:GenericTransformer=None) -> Tuple[List[str], np.array]:
        scores = list(score_conf.keys())
        score_weights = []
        for score_id,score_program in score_conf.items():
            if score_program["agg"] == "avg":
                score_weight,score = self._avg(score_id, score_program, df, None, transform)
            if score_program["agg"] == "weighted-avg":
                score_weight,score = self._avg(score_id, score_program, df, weights, transform)
            if score_program["agg"] == "frac":
                score_weight,score = self._frac(score_id, score_program, df, weights, transform)
            score_weights.append(score_weight)
            df.loc[:,score_id] = score
        return scores,np.array(score_weights)
