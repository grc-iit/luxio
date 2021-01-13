
from external_clients.json_client import JSONClient
from utils.mapper_manager import MapperManager
from common.configuration_manager import *
from common.enumerations import *
from io_requirement_extractor.trace_parser.trace_parser_factory import *
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

        #Acquire darshan trace data
        self._initialize()
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        input = darshan_parser.parse()

        #Create derivative features

        #Select relevant features
        #feature_path = os.path.join(self.conf.app_behavior_model_dir, "features")
        #features = pd.read_csv()

        #Map features to every class of I/O behavior with certain confidence

        # load sample/io_req_output.json into output
        output = JSONClient().load(self.conf.io_req_out_path)
        mapper = MapperManager()
        mapper.run(input, output)
        self._finalize()
        return output

    def _finalize(self) -> None:
        pass
