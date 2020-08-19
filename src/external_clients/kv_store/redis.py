from external_clients.kv_store.kv_store import KVStore
from common.configuration_manager import ConfigurationManager
from common.error_codes import *

import typing
import redis

'''
The implementation of redis database
'''
class RedisDB(KVStore):
    def __init__(self) -> None:
        super().__init__()
        conf = ConfigurationManager.get_instance()
        self.database = redis.Redis(host=conf.db_addr, port=conf.db_port, db=conf.db)

    def _put_impl(self, key: str, value: str) -> bool:
        try:
            return self.database.set(key, value)
        except:
            raise Error(ErrorCode.REDISDB_STORE_ERROR).format("RedisDB")

    def _get_impl(self, key: str) -> str:
        try:
            return self.database.get(key)
        except:
            raise Error(ErrorCode.REDISDB_GET_ERROR).format("RedisDB")

    def _query_impl(self, key: str) -> bool:
        try:
            return self.database.exists(key)
        except:
            raise Error(ErrorCode.REDISDB_QUERY_ERROR).format("RedisDB")
