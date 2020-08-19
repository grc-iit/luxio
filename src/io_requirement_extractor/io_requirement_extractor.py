from external_clients.json_client import JSONClient
from utils.mapper_manager import MapperManager
from common.configuration_manager import *
from common.enumerations import *
from database.database import *
from io_requirement_extractor.trace_parser.trace_parser_factory import *

class IORequirementExtractor:
    def __init__(self):
        self.conf = None

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()

    def run(self) -> dict:
        self._initialize()
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        input = darshan_parser.parse()
        job_spec = JSONClient().load(self.conf.job_spec)
        db = DataBase.get_instance()

        try:
            return DataBase.get(job_spec)
        except:
            pass

        # load sample/io_req_output.json into output
        output = JSONClient().load(self.conf.io_req_out_path)
        mapper = MapperManager()
        mapper.run(input, output)
        # call to database to store it key:input, val:output
        db.put(input, output)
        self._finalize()
        return output

    def _finalize(self) -> None:
        pass
