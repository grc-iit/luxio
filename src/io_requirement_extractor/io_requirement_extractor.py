from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager
from src.common.configuration_manager import *
from src.database.database import *

class IORequirementExtractor:
    def __init__(self):
        pass

    def _initialize(self) -> None:
        pass

    def run(self) -> dict:
        self._initialize()
        conf = ConfigurationManager.get_instance()

        #set the value from parser into input.
        # TODO: neeraj set the input from darshan parser
        # load sample/darshan.json into input
        input = JSONClient().load(conf.darshan_trace_path)

        # call to database to check if key:input exists if true skip mapping
        db = DataBase.get_instance()
        try:
            return db.get(input)
        except:
            pass

        # load sample/io_req_output.json into output
        output = JSONClient().load(conf.io_req_path) #TODO: what about schema verification
        mapper = MapperManager()
        mapper.run(input, output)
        # call to database to store it key:input, val:output
        db.put(input, output)
        self._finalize()
        
        return output

    def _finalize(self) -> None:
        pass

