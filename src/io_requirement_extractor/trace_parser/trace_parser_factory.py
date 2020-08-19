
from common.enumerations import *
from common.error_codes import *
from io_requirement_extractor.trace_parser.trace_parser import *
from io_requirement_extractor.trace_parser.darshan_trace_parser import *
from type import Type


class TraceParserFactory:

    DARSHAN = 'darshan'

    _parse_classes = {
        DARSHAN: DarshanTraceParser,
    }

    def __init__(self):
        pass

    def get(self, parser_id: int) -> Type[TraceParser]:
        try:
            return TraceParserFactory._parse_classes[parser_id]()
        except:
            raise Error(ErrorCode.PARSER_ID).format(parser_id)
