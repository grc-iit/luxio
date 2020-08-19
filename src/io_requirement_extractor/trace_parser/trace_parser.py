import abc


class TraceParser(abc.ABC):

    @abc.abstractmethod
    def parse(self, file_: str) -> None:
        """
        Parse a Trace and return extracted variables
        """
        pass
