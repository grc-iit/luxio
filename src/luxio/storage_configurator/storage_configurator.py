
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
from luxio.resolver import Resolver
import pandas as pd
import numpy as np
import configparser


class StorageConfigurator(ABC):
    """
    A class used to generate the storage configuration according to the
    given storage requirement
    """

    def __init__(self, file_='storage.conf') -> None:
        self.load_configs(file_)
        self.resolver = Resolver()

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        pass

    def get_cost(self, deployment_conf: dict) -> float:
        cost = 0
        for k, v in deployment_conf.items():
            cost += (self.current_price[k] * v)
        return cost

    def load_configs(self, file_) -> None:
        config = configparser.ConfigParser()
        config.read(file_)
        sections = config.sections()

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

    def price_function(self, tier: str):
        return self.naive(tier) * self.logistic(tier)

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

        # Acquire the set of available resources from the scheduler
        # Determine whether or not the qosas in the storage requirement can be satisfied, and if so, how much it costs
        # We will have to have some configuration file to relate features from the hardware with the benchmarks
        # Choose the lowest-price QoSA that can be satisfied
        self._finalize()
        return None

    def run(self, storage_requirement: pd.DataFrame) -> dict:
        """
        Mapping the storage requirement to its corresponding storage
        configuration.
        And then return the storage configuration.
        :param storage_requirement: dict
        :return: dict
        """
        self._initialize()
        # Acquire the set of available resources from the scheduler
        # Determine whether or not the qosas in the storage requirement can be
        # satisfied, and if so, how much it costs
        # Choose the lowest-price QoSA that can be satisfied
        self._finalize()
        return None

    @abstractmethod
    def load_json(self) -> dict:
        """
        Load json file and return a python dictionary
        :return: dict
        """
        pass
