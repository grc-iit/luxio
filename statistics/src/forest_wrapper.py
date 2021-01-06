from generic_model import GenericModel

class ForestWrapper(GenericModel):
    def __init__(self, model):
        self.model_ = model
        self.feature_importances_ = None

    def feature_importances(self):
        self.feature_importances_ = self.model_.feature_importances
