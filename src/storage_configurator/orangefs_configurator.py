
from common.error_codes import *
from common.configuration_manager import *
from storage_configurator.storage_configurator import *

class OrangefsConfigurator(StorageConfigurator):
    """
    A class used to generate the Orangefs storage configuration
    """
    def __init__(self):
        self.conf = None

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        if (self.conf.time_ops):
            self.start_time = time.time()

    def load_json(self):
        """
        Load Orangefs configuration json file and return a python dictionary
        :return: dict
        """
        return JSONClient().load(self.conf.storage_req_config_out_path)

    def _finalize(self) -> None:
        if (self.conf.time_ops):
            print(f"OrangeFS configurator time: {time.time() - self.start_time}")

