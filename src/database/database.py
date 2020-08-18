from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.common.enumerations import KVStoreType

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
            raise Exception("The DataBase class is a singleton!")
        else:
            DataBase._instance = self
        self.kvstore = None

    def _initialize(self) -> None:
        self.kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        print("init kvstore object: ", self.kvstore)

    def _finalize(self) -> None:
        del self.kvstore
        self.kvstore = None
        print(self.kvstore)

    def put(self, key, value):
        self.kvstore.put(key, value)

    def get(self, key):
        return self.kvstore.get(key)

    def query(self, key):
        return self.kvstore.query(key)

