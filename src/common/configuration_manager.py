from common.enumerations import *
from common.error_codes import *

class ConfigurationManager:
    _instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigurationManager._instance is None:
            ConfigurationManager()
        return ConfigurationManager._instance

    def __init__(self):
        if ConfigurationManager._instance is not None:
            raise Error(ErrorCode.TOO_MANY_INSTANCES).format("ConfigurationManager")
        else:
            ConfigurationManager._instance = self

        self.darshan_trace_path = None
        self.io_req_out_path = None
        self.storage_req_out_path = None
        self.storage_req_config_out_path = None
        self.storage_configurator_type = StorageConfiguratorType.ORANGEFS
        self.db_type = KVStoreType.REDIS
        self.db_addr = "127.0.0.1"
        self.db_port = "6379"
        self.db = None
        self.serializer_type = SerializerType.PICKLE

    """
    Configuration Varibales go here.
    """

    @staticmethod
    def load(filename):
        with open(filename) as fp:
            dict = json.load(fp)
        conf = ConfigurationManager.get_instance()
        conf.darshan_trace_path = dict["darshan_trace_path"]
        conf.io_req_out_path = dict["io_req_out_path"]
        conf.storage_req_out_path = dict["storage_req_out_path"]
        conf.storage_req_config_out_path = dict["storage_req_config_out_path"]
        conf.storage_configurator_type = dict["storage_configurator_type"]
        conf.db_type = dict["db_type"]
        conf.db_addr = dict["db_addr"]
        conf.db_port = dict["db_port"]
        conf.serializer_type = dict["serializer_type"]
        return conf