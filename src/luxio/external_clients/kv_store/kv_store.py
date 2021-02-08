
from abc import ABC, abstractmethod
from luxio.external_clients.serializer.serializer_factory import *
from luxio.common.configuration_manager import *

class KVStore(ABC):
    """
    A class used to provide interfaces to key-value store.
    1) Provides access to database to store, extract and query previously I/O requirements
    2) Provides access to database to store, extract and query previously storage requirements
    """
    def __init__(self):
        conf = ConfigurationManager.get_instance()
        self.serializer = SerializerFactory.get(conf.serializer_type)

    def put(self, key, value) -> None:
        """
        Store the given (key, value) pair into the key-value store
        :param key: dict
        :param value: dict
        """
        serialized_key = self.serializer.serialize(key)
        serialized_value = self.serializer.serialize(value)
        self._put_impl(serialized_key,serialized_value)

    def get(self, key) -> dict:
        """
        Get data from the key-value store by the given key
        :param key: dict
        :return: dict
        """
        serialized_key = self.serializer.serialize(key)
        serialized_value = self._get_impl(serialized_key)
        if serialized_value is None:
            return None
        else:
            return self.serializer.deserialize(serialized_value)

    def query(self, key) -> bool:
        """
        Query whether the given key exists in the key-value store
        :param key: dict
        :return: bool
        """
        serialized_key = self.serializer.serialize(key)
        return self._query_impl(serialized_key)

    @abstractmethod
    def _put_impl(self, key: str, value: str) -> bool:
        """
        Store the given (key, value) pair into the key-value store
        :param key: str
        :param value: str
        :return: bool
        """
        pass

    @abstractmethod
    def _get_impl(self, key: str) -> str:
        """
        Get data from the key-value store by the given key
        :param key: str
        :return: str
        """
        pass

    @abstractmethod
    def _query_impl(self, key: str) -> bool:
        """
        Query whether the given key exists in the key-value store
        :param key: str
        :return: bool
        """
        pass
