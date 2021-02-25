
import asyncio
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient


class Resolver(ABC):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()

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

    async def get(self, tier: str = ''):
        """
        returns the available resources from the scheduler
        if None specified returns entire dict
        since we are making query async this can be too right?
        """
        pass

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
