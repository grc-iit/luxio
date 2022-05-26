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

    def _download_requirements(self):
        self.conf.timer.resume()
        print("Can check DB")
        db = DataBase.get_instance()
        print("Checking cached results from database")
        req_dict = db.get(self.conf.job_spec)
        if req_dict is None:
            io_identifier, ranked_sslos = self._extract_requirements(db)
        else:
            print("Using cached results from database")
            io_identifier = req_dict["io"]
            ranked_sslos = req_dict["storage"]
        self.conf.timer.pause().log("CheckDB")
        return io_identifier, ranked_sslos

    def _extract_requirements(self, db=None):
        print("Extracting I/O requirements")
        #Extract I/O requirements
        extractor = IORequirementExtractor()
        io_identifier = extractor.run()
        ranked_sslos = None
        #The user has passed a sslo, job_id, or deployment directly
        if "sslo_id" in self.conf.job_spec or "job_id" in self.conf.job_spec or "deployment_id" in self.conf.job_spec:
            if self.conf.check_db:
                self.conf.timer.resume()
                db.put(self.conf.job_spec, {"io": io_identifier, "storage": None})
                self.conf.timer.pause().log("PutReqsToDB")
        #Map I/O requirement to sslos
        else:
            mapper = Mapper()
            ranked_sslos = mapper.run(io_identifier)
            if self.conf.check_db:
                self.conf.timer.resume()
                db.put(self.conf.job_spec, {"io": io_identifier, "storage": ranked_sslos})
                self.conf.timer.pause().log("PutReqsToDB")
        return io_identifier, ranked_sslos

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

        self.conf.timer.resume()
        self.conf.job_spec = JSONClient().load(self.conf.job_spec_path)
        self.conf.timer.pause().log("LoadJobSpec")

        io_identifier = None
        ranked_sslos = None

        #Extract or download I/O requirements
        if self.conf.check_db:
            io_identifier, ranked_sslos = self._download_requirements()
        else:
            io_identifier, ranked_sslos = self._extract_requirements()

        #Identify candidate deployments
        if 'job_id' not in self.conf.job_spec:
            resolver = Resolver()
            deployments = resolver.run(io_identifier, ranked_sslos)

        #Schedule the job
        scheduler = LuxioSchedulerClient()
        scheduler.schedule(self.conf.job_spec, deployments)

        self._finalize()
        return self.conf.job_spec

    def _finalize(self) -> None:
        if self.conf.timer_log_path:
            self.conf.timer.save(self.conf.timer_log_path)
