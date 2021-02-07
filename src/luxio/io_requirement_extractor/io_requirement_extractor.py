
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
from luxio.common.configuration_manager import *
from luxio.common.enumerations import *
from luxio.io_requirement_extractor.trace_parser.trace_parser_factory import *
import os

class IORequirementExtractor:
    """
    Extract the i/o requirement for Luxio
    """
    def __init__(self):
        self.conf = None

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()

    def run(self) -> dict:
        """
        Reading the trace input and then mapping it to the corresponding io requirement.
        It will return the io requirement
        :return: dict
        """

        self._initialize()

        #Acquire historical trace data
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        input = darshan_parser.parse()
        print(input)

        #Load I/O behavior classifier model
        #classifier = BehaviorClassifier.load(self.conf.app_behavior_classifier_path)

        #Feature extraction

        #Return the I/O Signature
        output = JSONClient().load(self.conf.io_req_out_path)
        mapper = MapperManager()
        mapper.run(input, output)
        self._finalize()
        return output

    def _finalize(self) -> None:
        pass
