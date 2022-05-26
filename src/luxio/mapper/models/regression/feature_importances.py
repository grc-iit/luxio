from luxio.mapper.models.common import *

class RFEFeatureImportances:
    def _per_model_importances(self, X, model):
        feature_importances = np.array(model.feature_importances_)
        feature_importances /= np.sum(feature_importances)
        return feature_importances