
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
import pandas as pd
import numpy as np
import configparser


class StorageConfigurator(ABC):
    """
    A class used to generate the storage configuration according to the given storage requirement
    """

    def __init__(self, file_ = 'storage.conf') -> None:
        self.load_configs(file_)

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        pass

    def load_configs(self, file_) -> None:
        config = configparser.ConfigParser()
        config.read(file_)
        l = config.sections()

        self.x = {}
        self.A = {}
        self.B = {}
        self.K = {}
        self.nu = {}
        self.Q = {}
        self.base_price = {}
        self.C = {}
        self.current_price = {}

        for i in l:
            self.A[i] = config.get(i, 'Lower_bound')
            self.B[i] = config.get(i, 'Growth_rate')
            self.K[i] = config.get(i, 'Upper_bound')
            self.nu[i] = 1.0
            self.Q[i] = 1.0
            self.C[i] = 1.0
            self.x[i] = 1
            self.current_price[i] = self.price_function(i)

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

    def tier_count(self, tier: str):
        """
        This function gets the number of remaining devices of specified tier
        currently just returns template val
        """
        return 10 if tier == 'TIER-1' else 20 if tier == 'TIER-2' else 30

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

    @abstractmethod
    def load_json(self) -> dict:
        """
        Load json file and return a python dictionary
        :return: dict
        """
        pass
