from src.generic_model import GenericModel

class ForestWrapper(GenericModel):
    def __init__(self, model):
        self.model = model
        self.feature_importances_ = None
        self.fitness_ = 0

    def calculate_importances_(self):
        self.feature_importances_ = self.model.feature_importances_
