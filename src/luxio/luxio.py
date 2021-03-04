from luxio.io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from luxio.storage_configurator.storage_configurator_factory import *
from luxio.storage_requirement_builder.storage_requirement_builder import *
from luxio.external_clients.json_client import *
from luxio.database.database import *

class LUXIO:
    def __init__(self):
        self.conf = ConfigurationManager.get_instance()
        pass

    def _initialize(self) -> None:
        pass

    def run(self) -> dict:
        """
        Run the luxio to get storage configuration.
        In this process, it will execute the following steps:
            1) Checking for the job_info in the database
            2) If the job_info isn't in the database, it will extract the i/o requirement and then build
            the storage requirement according to the i/o requirement
            3) If the job_info is in the database, then getting the i/o requirement and storage requirement
            from the database
            4) Getting the storage configuration by the storage requirement obtained from step 2) or 3)
        :return: dict
        """
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
            # run storage requirement builder
            builder = StorageRequirementBuilder()
            storage_requirement = builder.run(io_requirement)
            # store the io requirement and storage requirement into database
            if self.conf.check_db:
                db.put(job_spec, {"io": io_requirement, "storage": storage_requirement})

        #run storage configurator
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
