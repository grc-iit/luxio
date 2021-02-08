
from luxio.common.error_codes import *
from luxio.common.configuration_manager import *
from luxio.storage_configurator.storage_configurator import *

class OrangefsConfigurator(StorageConfigurator):
    """
    A class used to generate the Orangefs storage configuration
    """
    def __init__(self):
        pass

    def load_json(self):
        """
        Load Orangefs configuration json file and return a python dictionary
        :return: dict
        """
        conf = ConfigurationManager.get_instance()
        return JSONClient().load(conf.storage_req_config_out_path)
