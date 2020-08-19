from enum import Enum

"""Key-value store types supported  Luxio"""
class KVStoreType(Enum):
    REDIS = "redis"

    def __str__(self):
        return self.value


class TraceParserType(Enum):
    """Trace Parser Types supported in Luxio"""
    DARSHAN = 'darshan'

    def __str__(self) -> str:
        return self.value
