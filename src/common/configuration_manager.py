from src.common.enumerations import *
from src.common.error_codes import *

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
        self.io_req_path = None
        self.db_type = KVStoreType.REDIS
        self.db_addr = "127.0.0.1"
        self.db_port = "6379"
        self.db = None
        self.serializer_type = SerializerType.PICKLE

    """
    Configuration Varibales go here.
    """
    def load(self) -> None:
        """
        This method loads up a json of configurations and sets it in the class.
        """
        pass