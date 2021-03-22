from luxio.io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from luxio.mapper.mapper import Mapper
from luxio.resolver.resolver import Resolver
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.external_clients.json_client import JSONClient
from luxio.database.database import *
from luxio.common.timer import Timer

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

        self.conf.job_spec = JSONClient().load(self.conf.job_spec_path)
        if self.conf.check_db:
            db = DataBase.get_instance()
        try:
            if self.conf.check_db:
                self.conf.timer.resume()
                req_dict = db.get(job_spec)
                io_identifier = req_dict["io"]
                ranked_qosas = req_dict["storage"]
                self.conf.timer.pause().log("CheckDB")
            else:
                raise 1
        except:
            # run io requirement extractor
            extractor = IORequirementExtractor()
            io_identifier = extractor.run()
            # map I/O requiremnt to QoSAs
            mapper = Mapper()
            ranked_qosas = mapper.run(io_identifier)
            # store the io requirement and ranked_qosas into database
            if self.conf.check_db:
                self.conf.timer.resume()
                db.put(job_spec, {"io": io_identifier, "storage": ranked_qosas})
                self.conf.timer.pause().log("PutReqsToDB")

        # Build and configure the qosas
        resolver = Resolver()
        deployments = resolver.run(io_identifier, ranked_qosas)

        # Schedule the job
        scheduler = LuxioSchedulerClient()
        scheduler.schedule(self.conf.job_spec, deployments)

        self._finalize()
        return self.conf.job_spec

    def _finalize(self) -> None:
        if self.conf.timer_log_path:
            self.conf.timer.save(self.conf.timer_log_path)
