from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.common.enumerations import KVStoreType

import typing

class DataBase:
    def __init__(self) -> None:
        self.kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)

    def put(self, key, value):
        self.kvstore.put(key, value)

    def get(self, key):
        return self.kvstore.get(key)

    def query(self, key):
        return self.kvstore.query(key)

