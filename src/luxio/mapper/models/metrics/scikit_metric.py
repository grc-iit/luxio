
import numpy as np
from typing import Union

class ScikitMetric:
    modes = {
        'avg' : 'uniform_average',
        'variance_weighted' : 'variance_weighted',
        'all': 'raw_values'
    }

    def _output_validator(self, pred_y:np.array, score:Union[float,np.array]) -> Union[float,np.array]:
        if self.mode == 'all' and pred_y.ndim == 1:
            return np.mean(score)
        return score
