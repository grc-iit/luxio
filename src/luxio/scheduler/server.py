#import daemon
#from luxio.utils.daemon import daemon
from luxio.database.database import *
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager
from .schedulers.scheduler_factory import *

class LuxioSchedulerServer:
    """
    Build the storage requirements for Luxio
    """
    def __init__(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        self.scheduler = SchedulerFactory.get(self.conf.scheduler_type)
        self.db = DataBase.get_instance()

    #TODO: Should be a daemon
    def event_based_query(self):
        self.scheduler.set_callbacks(self)
        self.refresh_resource_graph()
        self.refresh_existing_deployments()
        while True:
            continue

    #TODO: Should be a daemon
    def periodic_query(self) -> None:
        while True:
            self.refresh_resource_graph()
            self.refresh_existing_deployments()
            time.sleep(self.conf.luxio_server_frequency)

    def refresh_resource_graph(self):
        self.db.put("resource_graph", self.scheduler.refresh_resource_graph())

    def refresh_existing_deployments(self):
        self.db.put("existing_deployments", self.scheduler.refresh_existing_deployments())

    def schedule(self, job_spec:dict):
        self.scheduler.schedule(job_spec)


def start_luxio_server():
    conf = ConfigurationManager.get_instance()
    if conf.luxio_server_mode == LuxioServerMode.EVENT_BASED:
        LuxioSchedulerServer().event_based_query()
    else:
        LuxioSchedulerServer().periodic_query()
