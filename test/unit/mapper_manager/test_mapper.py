import unittest, pytest

from luxio.utils.mapper_manager import MapperManager


class MyTestCase(unittest.TestCase):
    def test_cases(self):
        input = {
            "a": 2,
            "b": 10
        }
        output = {}
        output["a_per"] = {
            "guard":"True",
            "dependencies": [],
            "include": "import numpy as np;",
            "guard": "True",
            "expr": "self.output['a_per']['val'] = self.input['a']/(self.input['a']+self.input['b'])",
            "val": 0
        }
        output["a_per_a"] = {
            "guard":"True",
            "dependencies": ["a_per"],
            "include": "import numpy as np;",
            "guard": "True",
            "expr": "self.output['a_per_a']['val']=self.output['a_per']['val']*100.0",
            "val": 0
        }
        mapper = MapperManager()
        mapper.run(input, output)


if __name__ == "__main__":
    unittest.main()
