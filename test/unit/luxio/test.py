import unittest
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from luxio import *

class TestLuxio(unittest.TestCase):
    def test_luxio(self):
        conf = ConfigurationManager.get_instance()
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.darshan_trace_path="sample/darshan_trace.json"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"
        tool = LUXIO()
        tool.run()

if __name__ == "__main__":
    unittest.main()