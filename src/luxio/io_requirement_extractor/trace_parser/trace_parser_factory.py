
from luxio.common.enumerations import TraceParserType
from luxio.common.error_codes import Error, ErrorCode
from luxio.io_requirement_extractor.trace_parser.darshan.darshan_trace_parser import DarshanTraceParser
from luxio.io_requirement_extractor.trace_parser.trace_parser import TraceParser


class TraceParserFactory:
    """
    Factory used for creating TracePaser object
    """
    def __init__(self):
        pass

    @staticmethod
    def get_parser(type: TraceParserType) -> TraceParser:
        """
        Return a TraceParser object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: TraceParserType
        :return: TraceParser
        """
        if type == TraceParserType.DARSHAN:
            return DarshanTraceParser()
        else:
            raise Error(ErrorCode.NOT_IMPLEMENTED)
