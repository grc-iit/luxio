
from luxio.external_clients.kv_store.kv_store import KVStore
from luxio.common.configuration_manager import ConfigurationManager
from luxio.common.error_codes import *
import typing
import redis

class RedisDB(KVStore):
    """
    Storing and Getting data into/from redis
    """
    def __init__(self) -> None:
        super().__init__()
        conf = ConfigurationManager.get_instance()
        self.database = redis.Redis(host=conf.db_addr, port=conf.db_port, db=conf.db)

    def _put_impl(self, key: str, value: str) -> bool:
        """
        Store the given (key, value) pair into redis
        :param key: str
        :param value: str
        :return: bool
        """
        try:
            return self.database.set(key, value)
        except:
            raise Error(ErrorCode.REDISDB_STORE_ERROR).format("RedisDB")

    def _get_impl(self, key: str) -> str:
        """
        Get data from redis by the given key
        :param key: str
        :return: str
        """
        try:
            return self.database.get(key)
        except:
            raise Error(ErrorCode.REDISDB_GET_ERROR).format("RedisDB")

    def _query_impl(self, key: str) -> bool:
        """
        Query whether the given key exists in redis
        :param key:
        :return: bool
        """
        try:
            return self.database.exists(key)
        except:
            raise Error(ErrorCode.REDISDB_QUERY_ERROR).format("RedisDB")
