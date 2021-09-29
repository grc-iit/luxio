
import unittest, pytest
from luxio.common.error_codes import *
from luxio.external_clients.json_client import *

class TestJsonClient(unittest.TestCase):
    def test(self):
        json_dict = JSONClient().load("sample/io_req.json")
        stripped_dict = JSONClient().strip(json_dict)
        JSONClient().dumps(json_dict)
        JSONClient().dumps(stripped_dict)

if __name__ == "__main__":
    unittest.main()
