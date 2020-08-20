from io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from storage_configurator.storage_configurator_factory import *
from storage_requirement_builder.storage_requirement_builder import *
from external_clients.json_client import *
from database.database import *

class LUXIO:
    def __init__(self):
        self.conf = ConfigurationManager.get_instance()
        pass

    def _initialize(self) -> None:
        pass

    def run(self) -> dict:
        self._initialize()

        job_spec = JSONClient().load(self.conf.job_spec)
        if self.conf.check_db:
            db = DataBase.get_instance()
        try:
            if self.conf.check_db:
                req_dict = db.get(job_spec)
                io_requirement = req_dict["io"]
                storage_requirement = req_dict["storage"]
            else:
                raise 1
        except:
            # run io requirement extractor
            extractor = IORequirementExtractor()
            io_requirement = extractor.run()
            JSONClient().dumps(JSONClient().strip(io_requirement)) #TODO: Remove/comment print statement
            #
            builder = StorageRequirementBuilder()
            storage_requirement = builder.run(io_requirement)
            JSONClient().dumps(JSONClient().strip(storage_requirement)) #TODO: Remove/comment print statement
            #
            if self.conf.check_db:
                db.put(job_spec, {"io": io_requirement, "storage": storage_requirement})

        configurator = StorageConfiguratorFactory.get(self.conf.storage_configurator_type)
        configuration = configurator.run(storage_requirement)
        self._finalize()
        return configuration

    def _finalize(self) -> None:
        pass


if __name__ == '__main__':
    """
    The main method to start the benchmark runtime.
    """
    tool = LUXIO()
    tool.run()
    exit(0)
