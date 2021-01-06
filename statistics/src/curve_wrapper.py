from src.generic_model import GenericModel
import numpy as np

class CurveWrapper(GenericModel):
    def __init__(self, model):
        self.model = model
        self.feature_importances_ = None
        self.fitness_ = 0

    def calculate_importances_(self):
        abscoeff = np.absolute(self.model.coef_)
        self.feature_importances_ = abscoeff / sum(abscoeff)
