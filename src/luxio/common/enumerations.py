from enum import Enum


class KVStoreType(Enum):
    """
    Key-value store types supported in Luxio
    """
    REDIS = "redis"

    def __str__(self):
        return self.value


class TraceParserType(Enum):
    """
    Trace Parser Types supported in Luxio
    """
    DARSHAN = 'darshan'
    ARGONNE = 'argonne'

    def __str__(self) -> str:
        return self.value


class SerializerType(Enum):
    """
    Serialization Libraries supported in Luxio
    """
    PICKLE = "pickle"
    MSGPACK = "message_pack"

    def __str__(self):
        return self.value


class StorageConfiguratorType(Enum):
    """
    Storage Configuarator Libraries supported in Luxio
    """
    ORANGEFS = "orangefs"

    def __str__(self):
        return self.value


class ResolverType(Enum):
    """
    Resolvers supported in Luxio
    """
    LUXIO = 'luxio'
