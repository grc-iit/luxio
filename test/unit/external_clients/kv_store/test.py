import unittest

from src.external_clients.kv_store.kv_store_factory import KVStoreFactory
from src.common.enumerations import KVStoreType

class KVStoreTestCase(unittest.TestCase):
    check_dict = {
        "c": 15,
        "d": 17
    }

    def testCasePut(self):
        input = {
            "a": 2,
            "b": 10
        }

        output = {}
        output["a_per"] = {
            "dependencies": [],
            "include": "import numpy as np;",
            "expr": "self.output['a_per']['val'] = self.input['a']/(self.input['a']+self.input['b'])"
        }
        output["a_per_a"] = {
            "dependencies": ["a_per"],
            "include": "import numpy as np;",
            "expr": "self.output['a_per_a']['val']=self.output['a_per']['val']*100.0",
            "val": None
        }

        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put(input, output)


    def testCaseGet(self):
        input = {
            "a": 1,
            "b": 2
        }

        output = {}
        output["a_per"] = {
            "dependencies": [],
            "include": "import numpy as np;",
            "expr": "self.output['a_per']['val'] = self.input['a']/(self.input['a']+self.input['b'])"
        }
        output["a_per_a"] = {
            "dependencies": ["a_per"],
            "include": "import numpy as np;",
            "expr": "self.output['a_per_a']['val']=self.output['a_per']['val']*100.0",
            "val": None
        }
        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put(input, output)
        result1 = kvstore.get(input)
        self.assertEqual(output, result1)
        result2 = kvstore.get(KVStoreTestCase.check_dict)
        self.assertEqual(None, result2)

    def testCaseQuery(self):
        input = {
            "a": 1,
            "b": 2
        }

        output = {}
        output["a_per"] = {
            "dependencies": [],
            "include": "import numpy as np;",
            "expr": "self.output['a_per']['val'] = self.input['a']/(self.input['a']+self.input['b'])"
        }
        output["a_per_a"] = {
            "dependencies": ["a_per"],
            "include": "import numpy as np;",
            "expr": "self.output['a_per_a']['val']=self.output['a_per']['val']*100.0",
            "val": None
        }

        kvstore = KVStoreFactory.get_kv_store(KVStoreType.REDIS)
        kvstore.put(input, output)
        result1 = kvstore.query(input)
        self.assertEqual(True, result1)
        result2 = kvstore.query(KVStoreTestCase.check_dict)
        self.assertEqual(False, result2)

if __name__ == "__main__":
    unittest.main()