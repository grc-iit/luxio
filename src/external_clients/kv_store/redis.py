from src.external_clients.kv_store.kv_store import KVStore
from src.common.configuration_manager import ConfigurationManager

import typing
import redis

'''
The implementation of redis database
'''
class RedisDB(KVStore):
    def __init__(self) -> None:
        super().__init__()
        #self.conf = ConfigurationManager.get_instance()
        #self.conf.load()
        self.host = "127.0.0.1"
        self.port = "6379"
        self.database = redis.Redis(host=self.host, port=self.port, db=0)

    def _put_impl(self, key: str, value: str) -> bool:
        return self.database.set(key, value)

    def _get_impl(self, key: str) -> str:
        return self.database.get(key).decode("utf-8")

    def _query_impl(self, key: str) -> bool:
        return self.database.exists(key)

