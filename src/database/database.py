from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.common.error_codes import *
from src.common.configuration_manager import ConfigurationManager

import typing

class DataBase(object):
    _instance = None

    @staticmethod
    def get_instance():
        """Static access method"""
        if DataBase._instance is None:
            DataBase()
        return DataBase._instance

    def __init__(self) -> None:
        super().__init__()
        """Virtually private constructor"""
        if DataBase._instance is not None:
            raise Error(ErrorCode.TOO_MANY_INSTANCES).format("DataBase")
        else:
            DataBase._instance = self
        self.kvstore = None

    def _initialize(self) -> None:
        conf = ConfigurationManager.get_instance()
        self.kvstore = KVStoreFactory.get_kv_store(conf.db_type)

    def _finalize(self) -> None:
        del self.kvstore
        self.kvstore = None

    def put(self, key, value):
        self.kvstore.put(key, value)

    def get(self, key):
        return self.kvstore.get(key)

    def query(self, key):
        return self.kvstore.query(key)

