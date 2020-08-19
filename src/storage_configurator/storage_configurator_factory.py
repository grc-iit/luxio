from common.enumerations import *
from common.error_codes import *
from storage_configurator.storage_configurator import *
from storage_configurator.orangefs_configurator import *

class StorageConfiguratorFactory:
    def __init__(self):
        pass

    @staticmethod
    def get(type: StorageConfiguratorType) -> StorageConfigurator:
        if type == StorageConfiguratorType.ORANGEFS:
            return OrangefsConfigurator()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("StorageConfiguratorFactory", type)
