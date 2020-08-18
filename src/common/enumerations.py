from enum import Enum

"""Key-value store types supported  Luxio"""
class KVStoreType(Enum):
    REDIS = "redis"

    def __str__(self):
        return self.value

