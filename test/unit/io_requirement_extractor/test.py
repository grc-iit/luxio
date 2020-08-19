from io_requirement_extractor.trace_parser.darshan_trace_parser import DarshanTraceParser
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from io_requirement_extractor.io_requirement_extractor import *
import unittest
import json
from typing import Dict


class DarshanTraceParserTest(unittest.TestCase):
    input_file = 'sample/sample.darshan'
    output_file = 'sample/sample.json'

    def get_parse(self, file_: str) -> Dict:
        darshan_parser = DarshanTraceParser()
        extracted_darshan_variables = darshan_parser.parse(self.input_file)
        return extracted_darshan_variables

    def get_output(self, file_: str) -> Dict:
        with open(self.output_file, 'r') as json_file:
            data = json.load(json_file)
        return data

    def parse_testcase(self) -> None:
        assert(self.get_output() == self.get_parse())


class TestIOequirementExtractor(unittest.TestCase):
    def test_redis_extract(self):
        conf = ConfigurationManager.get_instance()
        conf.io_req_out_path="sample/io_req_output.json"
        conf.darshan_trace_path="sample/darshan_trace.json"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr = "127.0.0.1"
        conf.db_port = "6379"

        io_req_extractor = IORequirementExtractor()
        io_req_extractor.run()

        input = JSONClient().load(conf.darshan_trace_path)
        db = DataBase.get_instance()
        output_db = db.get(input)


if __name__ == "__main__":
    unittest.main()
