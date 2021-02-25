
import asyncio
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.external_clients.kv_store.kv_store_factory import KVStoreType
from luxio.common.enumerations import KVStoreType
from configparser import ConfigParser


class Resolver(ABC):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()
        self.db = KVStoreType(KVStoreType.REDIS)
        conf = ConfigParser()
        conf.read('storage.conf')
        self.counts = {}
        for i in conf.sections():
            val = conf.get(i, 'counts')
            self.counts[i] = val
            self.db._put_impl(i, val)

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        self.loop.close()

    def run(self) -> dict:
        """
        Query the available resources from the scheduler
        :param : dict
        :return: dict
        """
        self._initialize()
        self._finalize()
        return None

        self._finalize()
        return None

    def sync_to_db(self) -> None:
        for k, v in self.counts.items():
            self.db._put_impl(k, v)

    def relinquish_resources(self, resources: dict) -> bool:
        for k, v in resources.items():
            self.counts[k] += resources[k]
        self.sync_to_db()

    def acquire_resources(self, resources: dict) -> bool:
        for k, v in resources.items():
            self.counts[k] -= resources[k]
        self.sync_to_db()

    def get(self, tier: str = ''):
        """
        returns the available resources from the scheduler
        if None specified returns entire dict
        since we are making query async this can be too right?
        """
        return self.counts[tier]

    async def query(self):
        """
        function to periodically query the scheduler
        and maintian a dict of available storage resources
        should be ASYNC
        """
        pass

    @abstractmethod
    def load_json(self) -> dict:
        """
        Load json file and return a python dictionary
        :return: dict
        """
        pass
