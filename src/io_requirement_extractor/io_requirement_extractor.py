
from external_clients.json_client import JSONClient
from utils.mapper_manager import MapperManager
from common.configuration_manager import *
from common.enumerations import *
from io_requirement_extractor.trace_parser.trace_parser_factory import *

class IORequirementExtractor:
    """
    Extract the i/o requirement for Luxio
    """
    def __init__(self):
        self.conf = None

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        if (self.conf.time_ops):
            self.start_time = time.time()

    def run(self) -> dict:
        """
        Reading the trace input and then mapping it to the corresponding io requirement.
        It will return the io requirement
        :return: dict
        """
        self._initialize()
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        input = darshan_parser.parse()

        # load resources/io_req_output.json into output
        output = JSONClient().load(self.conf.io_req_out_path)
        mapper = MapperManager()
        mapper.run(input, output)
        self._finalize()
        return output

    def _finalize(self) -> None:
        if (self.conf.time_ops):
            print(f"I/O req extractor time: {time.time() - self.start_time}")
