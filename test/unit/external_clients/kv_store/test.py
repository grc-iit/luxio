import unittest

from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.external_clients.kv_store.kv_store import KVStore
from src.external_clients.kv_store.redis import RedisDB
from src.database.database import DataBase
from src.common.enumerations import KVStoreType

class KVStoreFactoryTestCase(unittest.TestCase):
    def testCase(self):
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)

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

class DataBaseTestCase(unittest.TestCase):
    def testCaseGetInstance(self):
        database = DataBase.get_instance();
        database1 = DataBase.get_instance();
        assert (database == database1)

if __name__ == "__main__":
    unittest.main()