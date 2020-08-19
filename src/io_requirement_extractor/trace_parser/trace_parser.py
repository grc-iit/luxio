from common.error_codes import ErrorCode, Error
import abc


class TraceParser(abc.ABC):

    @abc.abstractmethod
    def parse(self, file_: str) -> None:
        """
        Parse a Trace and return extracted variables
        """
        raise Error(ErrorCode.NOT_IMPLEMENTED).format(file_))
