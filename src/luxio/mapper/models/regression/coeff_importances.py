import numpy as np
from luxio.mapper.models.common import *

class RFECoeffImportances:
    def _per_model_importances(self, X, model):
        abscoeff = np.absolute(model.coef_)
        if abscoeff.ndim > 1:
            abscoeff = np.sum(abscoeff, axis=0)
        feature_importances = abscoeff / np.sum(abscoeff)
        return feature_importances