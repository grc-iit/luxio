import unittest
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from io_requirement_extractor.io_requirement_extractor import *

class TestIOequirementExtractor(unittest.TestCase):
    def test_redis_extract(self):
        conf = ConfigurationManager.get_instance()
        conf.io_req_path="sample/io_req_output.json"
        conf.darshan_trace_path="sample/darshan_trace.json"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr="127.0.0.1"
        conf.db_port="6379"

        io_req_extractor = IORequirementExtractor()
        io_req_extractor.run()

        input = JSONClient().load(conf.darshan_trace_path)
        db = DataBase.get_instance()
        output_db = db.get(input)

if __name__ == "__main__":
    unittest.main()