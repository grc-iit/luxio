
from luxio.external_clients.kv_store.redis import RedisDB
from luxio.external_clients.kv_store.kv_store import KVStore
from luxio.common.enumerations import KVStoreType
from luxio.common.error_codes import *
import typing

class KVStoreFactory(object):
    """
    Factory used for creating KVStore object
    """
    def __init__(self):
        pass

    @staticmethod
    def get_kv_store(type: KVStoreType) -> KVStore:
        """
        Return a KVStore object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: KVStoreType
        :return: KVStore
        """
        if type == KVStoreType.REDIS:
            return RedisDB()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("KVStoreFactory", type)
