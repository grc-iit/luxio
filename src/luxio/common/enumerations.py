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
    SCS_STRESS_TEST = 'scs_stress_test'
    SCS_QOSA = 'scs_qosa'

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
    REDIS = "redis"
    LABIOS = "labios"

    def __str__(self):
        return self.value

class ResolverPolicyType(Enum):
    """
    Resolvers policies supported in Luxio
    """
    MAX_PERFORMANCE = 'max_performance'
    MAX_UTILIZATION = 'max_utilization'
    MIN_INTERFERENCE = 'min_interference'

    def __str__(self):
        return self.value

class SchedulerType(Enum):
    """
    Schedulers supported in Luxio
    """
    DUMMY = 'dummy'

    def __str__(self):
        return self.value

class PriceType(Enum):
    """
    Price models supported in Luxio
    """
    AVAILABILITY = 'availability'
    DEMAND = 'demand'

    def __str__(self):
        return self.value

class DeploymentStatus(Enum):
    """
    The status of different deployments for the resolver
    """

    NEW = 0
    RUNNING = 1
    WAITING = 2

    def __str__(self) -> str:
        return str(self.value)

class LuxioServerMode(Enum):
    """
    The status of different deployments for the resolver
    """

    EVENT_BASED = 'event_based'
    PERIODIC = 'periodic'

    def __str__(self) -> str:
        return str(self.value)
