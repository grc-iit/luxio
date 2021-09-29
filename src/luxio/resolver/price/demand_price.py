
from .price import Price
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.external_clients.json_client import JSONClient
from luxio.external_clients.kv_store.kv_store_factory import KVStoreType
from luxio.common.enumerations import KVStoreType
from luxio.common.configuration_manager import ConfigurationManager

import pandas as pd
import numpy as np

class DemandPrice(Price):
    """
    Compute the demand price of a deployment
    """

    def __init__(self):
        self.load_configs(file_)

    def load_configs(self, file_) -> None:
        conf = ConfigurationManager.get_instance()
        price_conf = configparser.ConfigParser()
        price_conf.read(conf.price_conf)
        sections = price_conf.sections()

        self.x = {}
        self.A = {}
        self.B = {}
        self.K = {}
        self.nu = {}
        self.Q = {}
        self.base_price = {}
        self.C = {}
        self.current_price = {}

        for i in sections:
            self.A[i] = config.get(i, 'Lower_bound')
            self.B[i] = config.get(i, 'Growth_rate')
            self.K[i] = config.get(i, 'Upper_bound')
            self.base_price = config.get(i, 'Base_price')
            self.nu[i] = 1.0
            self.Q[i] = 1.0
            self.C[i] = 1.0
            self.x[i] = 1
            self.current_price[i] = self.price_function(i)

    def tier_count(self, tier: str):
        """
        This function gets the number of remaining devices of specified tier
        currently just returns template val
        """
        return self.resolver.get(tier)

    def naive(self, tier: str):
        return self.base_price[tier] / self.tier_count(tier)

    def logistic(self, tier: str):
        return (
            self.A[tier] + (
                (
                    self.K[tier] - self.A[tier]
                ) / (
                    self.C[tier] + (
                        self.Q[tier] * np.exp(-self.B[tier] * self.x[tier])
                    )
                ) ** (
                    1 / self.nu[tier]
                )
            )
        )

    def price_function(self, tier: str):
        return self.naive(tier) * self.logistic(tier)

    def price(self, deployments:pd.DataFrame, scheduler:LuxioSchedulerClient) -> pd.DataFrame:
        cost = 0
        for tier in scheduler.resource_graph.keys():
            cost += (self.current_price[tier] * self.price_function(tier))
        return cost
