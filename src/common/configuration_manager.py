
from common.enumerations import *
from common.error_codes import *
import json

class ConfigurationManager:
    """
    A singleton class used to provide configuration variables for luxio
    """
    _instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigurationManager._instance is None:
            ConfigurationManager()
        return ConfigurationManager._instance

    def __init__(self):
        """
        Initialize the configuration variables
        """
        if ConfigurationManager._instance is not None:
            raise Error(ErrorCode.TOO_MANY_INSTANCES).format("ConfigurationManager")
        else:
            ConfigurationManager._instance = self

        self.job_spec = None
        self.darshan_trace_path = None
        self.io_req_out_path = None
        self.storage_req_out_path = None
        self.storage_req_config_out_path = None
        self.storage_configurator_type = StorageConfiguratorType.ORANGEFS
        self.check_db = True
        self.db_type = KVStoreType.REDIS
        self.db_addr = "localhost"
        self.db_port = "6379"
        self.db = None
        self.serializer_type = SerializerType.PICKLE

    @staticmethod
    def load(filename):
        """
        Read configuration info from the given json file.
        :param filename: str
        :return: ConfigurationManager
        """
        with open(filename) as fp:
            dict = json.load(fp)
        conf = ConfigurationManager.get_instance()
        conf.job_spec = dict["job_spec_path"]
        conf.darshan_trace_path = dict["darshan_trace_path"]
        conf.io_req_out_path = dict["io_req_out_path"]
        conf.storage_req_out_path = dict["storage_req_out_path"]
        conf.storage_req_config_out_path = dict["storage_req_config_out_path"]
        conf.storage_configurator_type = dict["storage_configurator_type"]
        conf.check_db = dict["check_db"]
        conf.db_type = dict["db_type"]
        conf.db_addr = dict["db_addr"]
        conf.db_port = dict["db_port"]
        conf.serializer_type = dict["serializer_type"]
        return conf
