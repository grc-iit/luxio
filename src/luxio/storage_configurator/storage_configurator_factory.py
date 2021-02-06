
from luxio.common.enumerations import *
from luxio.common.error_codes import *
from luxio.storage_configurator.storage_configurator import *
from luxio.storage_configurator.orangefs_configurator import *

class StorageConfiguratorFactory:
    """
    Factory used for creating StorageConfigurator object
    """
    def __init__(self):
        pass

    @staticmethod
    def get(type: StorageConfiguratorType) -> StorageConfigurator:
        """
        Return a StorageConfigurator object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: StorageConfiguratorType
        :return: StorageConfigurator
        """
        if type == StorageConfiguratorType.ORANGEFS:
            return OrangefsConfigurator()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("StorageConfiguratorFactory", type)
