
from .resolver_policy import ResolverPolicy
from .max_performance import MaxPerformance
from .max_utilization import MaxUtilization
from .min_interference import MinInterference

from luxio.common.enumerations import *
from luxio.common.error_codes import *
import pandas as pd

class ResolverPolicyFactory:
    """
    Factory used for creating ResolverFactory object
    """

    def __init__(self):
        pass

    @staticmethod
    def get(type: ResolverPolicyType) -> ResolverPolicy:
        """
        Return a ResolverPolicyFactory object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: ResolverPolicyType
        :return: ResolverPolicy
        """
        if type == ResolverPolicyType.MAX_PERFORMANCE:
            return MaxPerformance()
        elif type == ResolverPolicyType.MAX_UTILIZATION:
            return MaxUtilization()
        elif type == ResolverPolicyType.MIN_INTERFERENCE:
            return MinInterference()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("ResolverPolicyFactory", type)
