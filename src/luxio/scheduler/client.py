
from luxio.external_clients.json_client import JSONClient
from luxio.database.database import *
from luxio.common.configuration_manager import *
from typing import List, Dict, Tuple
import pandas as pd

from .schedulers.scheduler_factory import *

class LuxioSchedulerClient:
    def __init__(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        self.db = DataBase.get_instance()
        self.resource_graph = None
        self.existing_deployments = None
        self.scheduler = SchedulerFactory.get(self.conf.scheduler_type)

    def download_resource_graph(self):
        self.resource_graph = self.db.get("resource_graph")
        return self.resource_graph

    def download_existing_deployments(self):
        self.existing_deployments = self.db.get("existing_deployments")
        return self.existing_deployments

    def remove_resources(self, resources:dict):
        for tier,count in resources.items():
            if self.resource_graph[tier] >= count:
                self.resource_graph[tier] -= count
            else:
                raise Error(ErrorCode.RESOURCES_UNAVAILABLE).format(count, tier, self.resource_graph[tier])

    def schedule(self, job_spec:dict, deployments:pd.DataFrame):
        for id,deployment in deployments.iterrows():
            job_spec['qosa'] = deployment
            if self.scheduler.schedule(job_spec):
                break
