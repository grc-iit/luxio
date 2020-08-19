from common.enumerations import *
from common.error_codes import *
from storage_configurator.storage_configurator import *
from storage_configurator.orangefs_configurator import *

class StorageConfiguratorFactory:
    def __init__(self):
        pass

    @staticmethod
    def get(storage_configurator_id: StorageConfiguratorType) -> StorageConfigurator:
        if storage_configurator_id == StorageConfiguratorType.ORANGEFS:
            return OrangefsConfigurator()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("StorageConfiguratorFactory", storage_configurator_id)
