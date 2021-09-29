
from luxio.common.enumerations import *
from luxio.common.error_codes import *

from .scheduler import Scheduler
from .dummy import DummyScheduler

class SchedulerFactory:
    """
    Factory used for creating ResolverFactory object
    """

    def __init__(self):
        pass

    @staticmethod
    def get(type: SchedulerType) -> Scheduler:
        """
        Return a SchedulerFactory object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: SchedulerFactory
        :return: Scheduler
        """

        if isinstance(type, str):
            type = SchedulerType[type]
        if type == SchedulerType.DUMMY:
            return DummyScheduler()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("SchedulerFactory", type)
