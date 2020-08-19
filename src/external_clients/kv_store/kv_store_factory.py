from external_clients.kv_store.redis import RedisDB
from external_clients.kv_store.kv_store import KVStore
from common.enumerations import KVStoreType

import typing

class KVStoreFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_kv_store(type: KVStoreType) -> KVStore:
        if type == KVStoreType.REDIS:
            return RedisDB()
        else:
            raise Exception(str("Invalid key-value store type"))

