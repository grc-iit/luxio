import unittest

from src.utils.mapper_manager import MapperManager


class MyTestCase(unittest.TestCase):
    def test_cases(self):
        input={
            "a": 2,
            "b":10
        }
        output={}
        output["a_per"]={
                "dependencies": [],
                "include": "import numpy as np;",
                "expr": "self.output['a_per']['val'] = self.input['a']/(self.input['a']+self.input['b'])"
            }
        output["a_per_a"] = {
            "dependencies" : ["a_per"],
            "include":"import numpy as np;",
            "expr": "self.output['a_per_a']['val']=self.output['a_per']['val']*100.0",
            "val":None
        }
        mapper = MapperManager()
        mapper.run(input,output)

