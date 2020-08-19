
from src.io_requirement_extractor.trace_parser.darshan_trace_parser import darshan_trace_parser
import unittest
import json
from typing import Dict


class DarshanTraceParserTest(unittest.TestCase):
    input_file = '../../../sample/sample.darshan'
    output_file = '../../../sample/sample.json'

    def get_parse(self, file_: str) -> Dict:
        darshan_parser = darshan_trace_parser()
        extracted_darshan_variables = darshan_parser.parse(self.input_file)
        return extracted_darshan_variables

    def get_output(self, file_: str) -> Dict:
        with open(self.output_file, 'r') as json_file:
            data = json.load(json_file)
        return data

    def parse_testcase(self) -> None:
       assert(self.get_output() == self.get_parse())
