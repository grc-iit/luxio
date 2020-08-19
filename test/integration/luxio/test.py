import unittest
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from luxio import *

class TestLuxio(unittest.TestCase):
    def test_luxio(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.darshan_trace_path="sample/sample.darshan"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.storage_req_config_out_path="sample/stor_req_conf_output.json"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        tool = LUXIO()
        config = tool.run()
        print(config)


if __name__ == "__main__":
    unittest.main()