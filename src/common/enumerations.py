from enum import Enum

"""Key-value store types supported  Luxio"""
class KVStoreType(Enum):
    REDIS = "redis"

"""Serialization Libraries supported"""
class SerializerType(Enum):
    PICKLE = "pickle"
    MSGPACK = "message_pack"
