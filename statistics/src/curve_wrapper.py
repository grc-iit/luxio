from generic_model import GenericModel

class CurveWrapper(GenericModel):
    def __init__(self, model):
        self.model_ = model
        self.feature_importances_ = None

    def feature_importances(self):
        abscoeff = np.absolute(model.coef_[0])
        self.feature_importances_ = abscoeff / sum(abscoeff)
