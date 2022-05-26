
import numpy as np

def _identity(X:np.array, axis=0) -> np.array:
    return X

def _variance_weighted(X:np.array, axis=0) -> np.array:
    return np.average(X, axis=axis, weights=np.nanvar(X,axis=1))

class CustomMetric:
    modes = {
        'min' : np.nanmin,
        'avg' : np.nanmean,
        'variance_weighted' : _variance_weighted,
        'max' : np.nanmax,
        'median' : np.nanmedian,
        'all' : _identity
    }
