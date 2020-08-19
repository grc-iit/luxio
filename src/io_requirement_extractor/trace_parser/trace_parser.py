
from common.error_codes import ErrorCode
from common.error_codes import Error
import abc


class TraceParser(abc.ABC):

    @abc.abstractmethod
    def parse(self) -> None:
        """
        Parse a Trace and return extracted variables
        """
        raise Error(ErrorCode.NOT_IMPLEMENTED).format(file_)
