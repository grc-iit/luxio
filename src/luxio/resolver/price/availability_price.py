
from .price import Price
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager
from luxio.common.enumerations import *

import pandas as pd
import numpy as np

class AvailabilityPrice(Price):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    def __init__(self):
        self.conf = ConfigurationManager.get_instance()
        self.tier_costs = JSONClient().load(self.conf.price_conf)

    def price(self, deployments:pd.DataFrame, scheduler:LuxioSchedulerClient) -> pd.DataFrame:
        rg = scheduler.resource_graph
        deployments["price"] = 0
        for tier,tier_cost in self.tier_costs.items():
            idx = deployments.status == DeploymentStatus.NEW
            resource_count = deployments[idx][tier].to_numpy()
            avail_resources = rg[tier] - resource_count
            deployments.loc[idx,"price"] += resource_count * (tier_cost / avail_resources)
        deployments[deployments.price < 0] = np.inf
        return deployments
