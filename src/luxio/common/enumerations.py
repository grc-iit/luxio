from enum import Enum

"""
Key-value store types supported in Luxio
"""
class KVStoreType(Enum):
    REDIS = "redis"

    def __str__(self):
        return self.value

"""
Trace Parser Types supported in Luxio
"""
class TraceParserType(Enum):
    DARSHAN = 'darshan'
    ARGONNE = 'argonne'

    def __str__(self) -> str:
        return self.value

"""
Serialization Libraries supported in Luxio
"""
class SerializerType(Enum):
    PICKLE = "pickle"
    MSGPACK = "message_pack"

    def __str__(self):
        return self.value

"""
Storage Configuarator Libraries supported in Luxio
"""
class StorageConfiguratorType(Enum):
    ORANGEFS = "orangefs"

    def __str__(self):
        return self.value

