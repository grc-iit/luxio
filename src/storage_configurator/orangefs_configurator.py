
from common.error_codes import *
from common.configuration_manager import *
from storage_configurator.storage_configurator import *

class OrangefsConfigurator(StorageConfigurator):
    def __init__(self):
        pass

    def load_json(self):
        conf = ConfigurationManager.get_instance()
        return JSONClient().load(conf.storage_req_config_out_path)