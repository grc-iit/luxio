import unittest, pytest
import json
from typing import Dict

from luxio.io_requirement_extractor.trace_parser.darshan_trace_parser import DarshanTraceParser
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.io_requirement_extractor.io_requirement_extractor import *
from luxio.database.database import *

class TestIORequirementExtractor(unittest.TestCase):
    def test_redis_extract(self):
        conf = ConfigurationManager.get_instance()
        conf.job_spec="sample/job_info.json"
        conf.io_req_out_path="sample/io_req_output.json"
        conf.darshan_trace_path="sample/sample.darshan"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr = "127.0.0.1"
        conf.db_port = "6379"
        io_req_extractor = IORequirementExtractor()
        io_req_extractor.run()

        job_spec = JSONClient().load(conf.job_spec)
        db = DataBase.get_instance()
        output_db = db.get(job_spec)

if __name__ == "__main__":
    unittest.main()
