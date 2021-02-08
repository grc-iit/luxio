import unittest, pytest
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.external_clients.json_client import *
from luxio.luxio import LUXIO

class TestLuxio(unittest.TestCase):

    def test_luxio_sample(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.darshan_trace_path="sample/sample.darshan"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.storage_req_config_out_path="sample/stor_req_conf_output.json"
        conf.check_db = False
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        tool = LUXIO()
        config = tool.run()

        JSONClient().dumps(JSONClient().strip(config))
        print()
        print()

    def test_luxio_vpic(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.darshan_trace_path="sample/vpic.darshan"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.storage_req_config_out_path="sample/stor_req_conf_output.json"
        conf.check_db = False
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        tool = LUXIO()
        config = tool.run()

        JSONClient().dumps(JSONClient().strip(config))
        print()
        print()

    def test_luxio_hacc_io_read(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.darshan_trace_path="sample/hacc_io_read.darshan"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.storage_req_config_out_path="sample/stor_req_conf_output.json"
        conf.check_db = False
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        tool = LUXIO()
        config = tool.run()

        JSONClient().dumps(JSONClient().strip(config))
        print()
        print()

    def test_luxio_hacc_io_write(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.darshan_trace_path="sample/hacc_io_write.darshan"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.storage_req_out_path="sample/stor_req_output.json"
        conf.storage_req_config_out_path="sample/stor_req_conf_output.json"
        conf.check_db = False
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        tool = LUXIO()
        config = tool.run()

        JSONClient().dumps(JSONClient().strip(config))
        print()
        print()

if __name__ == "__main__":
    unittest.main()
