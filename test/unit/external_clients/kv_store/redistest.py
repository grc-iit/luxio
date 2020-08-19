import unittest
from src.external_clients.kv_store.redis import RedisDB

class RedisDBTestCase(unittest.TestCase):
    def testCasePut(self):
        redisdb = RedisDB()
        result = redisdb._put_impl("name", "haha")
        self.assertEqual(True, result)

    def testCaseGet(self):
        redisdb = RedisDB()
        testkey = "name"
        testvalue = "yejie"
        redisdb._put_impl(testkey, testvalue)
        value = redisdb._get_impl(testkey)
        self.assertEqual(testvalue, value.decode("utf-8"))
        value = redisdb._get_impl("age")
        self.assertEqual(None, value)

    def testCaseQuery(self):
        redisdb = RedisDB()
        redisdb._put_impl("name", "haha")
        testkey1 = "age"
        result = redisdb._query_impl(testkey1)
        self.assertEqual(False, result)
        testkey2 = "name"
        result = redisdb._query_impl(testkey2)
        self.assertEqual(True, result)

if __name__ == "__main__":
    unittest.main()
