
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Dict

def _listify(data):
    if not isinstance(data, list):
        return [data]
    return data

def load_features(path:str) -> List[str]:
    return list(pd.read_csv(path, header=None).iloc[:,0])

def _basic_stats(df:pd.DataFrame, n:int, metric:str='all') -> Dict[str,float]:
    metrics = {
        "n": len(df),
        "rel_n": len(df)/n,
        "mean": df.mean(),
        "std": df.std(),
        "0%": df.min(),
        "25%": df.quantile(q=.25),
        "50%": df.median(),
        "75%": df.quantile(q=.75),
        "90%": df.quantile(q=.9),
        "100%": df.max()
    }
    if metric == 'all':
        return metrics
    else:
        return metrics[metric]

def analyze_df(df:pd.DataFrame, features:List[str]=None, metric:str='all', n:int=None) -> Dict[str,float]:
    if features == None:
        features = df.columns
    else:
        features = _listify(features)
    if n is None:
        n = len(df)
    return { feature: _basic_stats(df[feature], n, metric=metric) for feature in features }

def random_sample(df:pd.DataFrame, w_train:float, w_hyper:float=0) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if isinstance(w_train,float):
        w_train = int(w_train*len(df))
    if isinstance(w_hyper,float):
        w_hyper = int(w_hyper*len(df))
    n = w_train + w_hyper

    if len(df) > n:
        sample = df.sample(n, replace=False)
        train = sample.iloc[0:w_train,:]
        hyper = sample.iloc[w_train:,:]
        test = df.drop(sample.index)
        return (train, hyper, test)
    else:
        return (df.sample(n, replace=True), df, df)

def make_mat(X:np.array) -> np.array:
    if X is None:
        return None
    X = np.array(X)
    if X.ndim < 2:
        X = np.reshape(X, (len(X), 1))
    return X

def make_vec(X:np.array) -> np.array:
    if X is None:
        return None
    X = np.array(X)
    if X.ndim > 1:
        X = X.flatten()
    return X

def assert_shape(v1:np.array, v2:np.array) -> None:
    if v1.shape != v2.shape:
        raise Exception(f"Numpy arrays do not have the same shape: {v1.shape} vs {v2.shape}")

def assert_axis(v1:np.array, v2:np.array, axis1=0, axis2=0) -> None:
    if v1.shape[axis1] != v2.shape[axis2]:
        raise Exception(f"Numpy arrays do not have the same axis shape: {v1.shape} vs {v2.shape} (axis1={axis1}, axis2={axis2})")
