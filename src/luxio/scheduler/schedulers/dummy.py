
from .scheduler import Scheduler
from luxio.external_clients.json_client import JSONClient
from luxio.common.enumerations import DeploymentStatus
from luxio.common.configuration_manager import ConfigurationManager
from typing import Dict
import pandas as pd

class DummyScheduler(Scheduler):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    initialized = False

    def __init__(self) -> None:
        self.resource_graph = None
        conf = ConfigurationManager.get_instance()
        self.resource_graph = JSONClient().load(conf.resource_graph_path)
        self.existing_deployments = pd.DataFrame()
        self.callbacks = None

    def set_callbacks(self, scheduler):
        self.callbacks = scheduler

    def refresh_resource_graph(self):
        return self.resource_graph

    def refresh_existing_deployments(self):
        return self.existing_deployments

    def schedule(self, job_spec:dict):
        if job_spec['qosa']['status'] == DeploymentStatus.RUNNING:
            self.existing_deployments[job_spec['qosa']['job_id']].append(job_spec)
        else:
            job_id = len(self.existing_deployments.keys())
            job_spec['qosa']['job_id'] = job_id
            job_spec['qosa']['status'] = DeploymentStatus.RUNNING
            self.existing_deployments.append(job_spec['qosa'])
        if self.callbacks is not None:
            self.callbacks.refresh_resource_graph()
            self.callbacks.refresh_existing_deployments()
        return True
