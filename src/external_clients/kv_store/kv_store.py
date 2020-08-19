from abc import ABC, abstractmethod
from external_clients.serializer.serializer_factory import *
from common.configuration_manager import *

class KVStore(ABC):
    """Interface to key-value store.
    1) Provides access database to store, extract and query previously I/O requirements
    2) Provides access database to store, extract and quey previously storage requirements
    """

    def __init__(self):
        conf = ConfigurationManager.get_instance()
        self.serializer = SerializerFactory.get(conf.serializer_type)

    def put(self, key, value):
        serialized_key = self.serializer.serialize(key)
        serialized_value = self.serializer.serialize(value)
        self._put_impl(serialized_key,serialized_value)

    def get(self, key):
        serialized_key = self.serializer.serialize(key)
        serialized_value = self._get_impl(serialized_key)
        return self.serializer.deserialize(serialized_value)

    def query(self, key):
        serialized_key = self.serializer.serialize(key)
        return self._query_impl(serialized_key)

    @abstractmethod
    def _put_impl(self, key: str, value: str) -> bool:
        pass

    @abstractmethod
    def _get_impl(self, key: str) -> str:
        pass

    @abstractmethod
    def _query_impl(self, key: str) -> bool:
        pass

