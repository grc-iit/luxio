
from .generic_transformer import GenericTransformer
from .log_transformer import LogTransformer
from .scale_transformer import ScaleTransformer


class TransformerFactory(GenericTransformer):
    methods = {
        "log10p1": "log10p1_",
        "log2p1": "log2p1_"
    }

    def __init__(self, method=None):
        super().__init__()
        self.method = method

    def set_method(self, method):
        self.method = method

    def _get_method(self):
        return self.__dict__[TransformerFactory.methods[self.method]]

    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        self.log10p1_ = LogTransformer(base=10, add=1)
        self.log2p1_ = LogTransformer(base=2, add=1)
        return self.transform(X)

    def inverse_transform(self,X):
        if self.method is None:
            return X
        return self._get_method().inverse_transform(X)

    def transform(self,X):
        if self.method is None:
            return X
        return self._get_method().transform(X)
