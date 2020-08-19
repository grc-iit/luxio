from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager


class IORequirementExtractor:
    def __init__(self):
        pass

    def _initialize(self) -> None:

        pass

    def run(self) -> dict:
        self._initialize()

        # set the value from parser into input.

        # TODO: neeraj set the input from darshan parser
        # load sample/darshan.json into input
        input_ = JSONClient().load(conf.darshan_trace_path)
        # call to database to check if key:input exists if true skip mapping
        # load sample/io_req_output.json into output
        output = JSONClient().load("io_requirement.json")
        mapper = MapperManager()
        mapper.run(input, output)
        # call to database to store it key:input, val:output
        self._finalize()
        return output

    def _finalize(self) -> None:
        pass
