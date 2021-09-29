
from .price import Price
from .availability_price import AvailabilityPrice
from .demand_price import DemandPrice
from luxio.common.enumerations import *
from luxio.common.error_codes import *
import pandas as pd

class PriceFactory:
    """
    Factory used for creating ResolverFactory object
    """

    def __init__(self):
        pass

    @staticmethod
    def get(type: PriceType) -> Price:
        """
        Return a PriceFactory object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: PriceType
        :return: Price
        """
        if type == PriceType.AVAILABILITY:
            return AvailabilityPrice()
        if type == PriceType.DEMAND:
            return DemandPrice()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("PriceFactory", type)
