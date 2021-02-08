import unittest, pytest

from luxio.database.database import DataBase

class DataBaseTestCase(unittest.TestCase):
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

        db = DataBase.get_instance()
        db._initialize()
        db.put(input, output)
        db._finalize()

    def testCaseGet(self):
        input = {
            "a": 100,
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

        db = DataBase.get_instance()
        db._initialize()
        db.put(input, output)
        result1 = db.get(input)
        self.assertEqual(output, result1)
        result2 = db.get(DataBaseTestCase.check_dict)
        self.assertEqual(None, result2)
        db._finalize()

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

        db = DataBase.get_instance()
        db._initialize()
        db.put(input, output)
        result1 = db.query(input)
        self.assertEqual(True, result1)
        result2 = db.query(DataBaseTestCase.check_dict)
        self.assertEqual(False, result2)
        db._finalize()

if __name__ == "__main__":
    unittest.main()
