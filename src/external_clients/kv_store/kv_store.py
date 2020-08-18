from abc import ABC, abstractmethod


class KVStore(ABC):
    """Interface to key-value store.
    1) Provides access database to store, extract and query previously I/O requirements
    2) Provides access database to store, extract and quey previously storage requirements
    """

    def __init__(self):
        serializer = None

    def put(self, key, value):
        serialized_key = serializer.serialize(key)
        serialized_value = serializer.serialize(value)
        self._put_impl(serialized_key,serialized_value)

    def get(self, key):
        serialized_key = serializer.serialize(key)
        serialized_value = self._get_impl(serialized_key)
        return serializer.deserialize(serialized_value)

    def query(self, key):
        serialized_key = serializer.serialize(key)
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

