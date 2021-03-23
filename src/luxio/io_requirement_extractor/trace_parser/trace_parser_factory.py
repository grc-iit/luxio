
from luxio.common.enumerations import TraceParserType
from luxio.common.error_codes import Error, ErrorCode
from luxio.io_requirement_extractor.trace_parser.darshan.dict import DarshanDictParser
from luxio.io_requirement_extractor.trace_parser.darshan.v3_2_1 import DarshanTraceParser_3_2_1
from luxio.io_requirement_extractor.trace_parser.darshan.anl import ArgonneTraceParser
from luxio.io_requirement_extractor.trace_parser.scs_qosa import SCSStressTestParser
from luxio.io_requirement_extractor.trace_parser.scs_qosa import SCSQosaParser
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
            return DarshanTraceParser_3_2_1()
        elif type == TraceParserType.DARSHAN_DICT:
            return DarshanDictParser()
        elif type == TraceParserType.ARGONNE:
            return ArgonneTraceParser()
        elif type == TraceParserType.SCS_STRESS_TEST:
            return SCSStressTestParser()
        elif type == TraceParserType.SCS_QOSA:
            return SCSQosaParser()
        else:
            raise Error(ErrorCode.NOT_IMPLEMENTED)
