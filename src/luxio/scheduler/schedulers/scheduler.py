
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager

class Scheduler(ABC):
    """
    A class used to Query the scheduler and keep track of existing resources
    """
    
    @abstractmethod
    def set_callbacks(self, scheduler):
        return

    @abstractmethod
    def refresh_resource_graph(self):
        return

    @abstractmethod
    def refresh_existing_deployments(self):
        return

    @abstractmethod
    def schedule(self, job_spec:dict):
        return
