import unittest

from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.external_clients.kv_store.kv_store import KVStore
from src.external_clients.kv_store.redis import RedisDB
from src.common.enumerations import KVStoreType

class KVStoreFactoryTestCase(unittest.TestCase):
    def testCase(self):
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)


class KVStoreTestCase(unittest.TestCase):
    def testCasePut(self):
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put()

    def testCaseGet(self):
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put()
        kvstore.get("name")

    def testCaseQuery(self):
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put()
        kvstore.query("name")

class RedisDBTestCase(unittest.TestCase):
    def testCasePut(self):
        redisdb = RedisDB()
        result = redisdb._put_impl("name", "haha")
        assert (result == True)

    def testCaseGet(self):
        redisdb = RedisDB()
        redisdb._put_impl("name", "jie ye")
        value = redisdb._get_impl("name")
        assert (value == "jie ye")

    def testCaseQuery(self):
        redisdb = RedisDB()
        redisdb._put_impl("name", "haha")
        result = redisdb._query_impl("age")
        assert (result == False)
        result = redisdb._query_impl("name")
        assert (result == True)

