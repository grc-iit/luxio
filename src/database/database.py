from common.configuration_manager import *
from external_clients.kv_store.kv_store_factory import KVStoreFactory
from common.enumerations import KVStoreType

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
        self._initialize()

    def __del__(self) -> None:
        self._finalize()

    def _initialize(self) -> None:
        self.kvstore = KVStoreFactory.get_kv_store(ConfigurationManager.get_instance().db_type)

    def _finalize(self) -> None:
        del self.kvstore
        self.kvstore = None

    def put(self, key, value) -> None:
        self.kvstore.put(key, value)

    def get(self, key) -> dict:
        return self.kvstore.get(key)

    def query(self, key) -> dict:
        return self.kvstore.query(key)

