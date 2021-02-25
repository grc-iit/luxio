
from luxio.common.enumerations import *
from luxio.common.error_codes import *


class ResolverFactory:
    """
    Factory used for creating ResolverFactory object
    """

    def __init__(self):
        pass

    @staticmethod
    def get(type: ResolverFactory) -> ResolverFactory:
        """
        Return a ResolverFactory object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: StorageConfiguratorType
        :return: StorageConfigurator
        """
        if type == ResolverType.LUXIO:
            return OrangefsConfigurator()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("StorageConfiguratorFactory", type)
