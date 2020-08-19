
from common.enumerations import TraceParserType
from common.error_codes import Error, ErrorCode
from io_requirement_extractor.trace_parser.darshan_trace_parser import DarshanTraceParser
from io_requirement_extractor.trace_parser.trace_parser import TraceParser
from type import Type


class TraceParserFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_parser(self, parser_id: str) -> Type[TraceParser]:
        if type == TraceParserType.DARSHAN:
            return DarshanTraceParser()
        else:
            raise Error(ErrorCode.NOT_IMPLEMENTED)
