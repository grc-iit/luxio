import darshan
from io_requirement_extractor.trace_parser.trace_parser import TraceParser
from typing import List, Dict, Tuple


class DarshanTraceParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    _extracted_variables = {}
    _formatted_variables = {}
    _sum_counters = [
        'total_reads',
        'total_writes'
        'total_bytes_read',
        'total_bytes_written',
        'read_time',
        'write_time',
    ]

    def __init__(self) -> None:
        pass

    def _get_max_byte_op(self) -> List[int]:
        """
        Get the max read and write bytes
        return: [max_read, max_write]
        """
        prefix = ['POSIX', 'STDIO']
        nome = '_MAX_BYTE_'
        read_ = []
        write_ = []
        for i in prefix:
            for j in self.dar_dict[i]:
                read_.append(j['counters'][f'{i}{nome}READ'])
                write_.append(j['counters'][f'{i}{nome}WRITTEN'])
        return [max(read_), max(write_)]

    def _get_total_module_stats(self, module_: str) -> List[float]:
        """
        Gets the total aggregated stats in Darshan
        return: [total_reads, total_writes, total_bytes_read, total_bytes_written,
        total_read_time_est, total_write_time_est]

        """
        module = 'MPIIO' if module_ == 'MPI-IO' else module_
        rw_prefixs = ['MPIIO_COLL', 'MPIIO_INDEP', 'MPIIO_SPLIT', 'MPIIO_NB'] \
            if module == 'MPIIO' else [module]
#        count_prefixs = ['MPIIO_COLL','MPIIO_INDEP'] if module == 'MPIIO' \
#            else [module]

        all_reads = []
        all_writes = []
#        all_count_files=[]
        all_bytes_read = []
        all_bytes_written = []
        all_read_time_est = []
        all_write_time_est = []
        for i in self.dar_dict[module_]:
            for j in rw_prefixs:
                all_reads.append(i['counters'][f'{j}_READS'])
                all_writes.append(i['counters'][f'{j}_WRITES'])

#            for j in count_prefixs:
#                all_count_files.append(i['counters'][f'{j}_OPENS'])

            all_bytes_read.append(i['counters'][f'{module}_BYTES_READ'])
            all_bytes_written.append(i['counters'][f'{module}_BYTES_WRITTEN'])
            all_read_time_est.append(i['fcounters'][f'{module}_F_READ_TIME'])
            all_write_time_est.append(i['fcounters'][f'{module}_F_WRITE_TIME'])
        return [
            sum(all_reads),
            sum(all_writes),
#            sum(all_count_files),
            sum(all_bytes_read),
            sum(all_bytes_written),
            sum(all_read_time_est),
            sum(all_write_time_est)
        ]

    def _output_formatter(self):
        """
        Converts the extracted variables into the specific format
        return: Dict[str, Dict[str, float]]
        """
        for k, v in self._extracted_variables.items():
            self._formatted_variables[k] = {'val': v}
        return self._formatted_variables

    def _get_max_op_time_size(self) -> List[Tuple[float, float]]:
        """
        Gets the max read bytes + time and max write bytes + time
        return: [(max_read_time, max_read_time_size), (max_write_time, max_write_time_size)]
        """
        modules = ['POSIX', 'MPI-IO', 'H5D']
        read_time = 0
        write_time = 0
        max_read = 0
        max_write = 0
        for i in modules:
            for j in self.dar_dict[i]:
                mod = 'MPIIO' if i == 'MPI-IO' else i
                if j['fcounters'][f'{mod}_F_MAX_READ_TIME'] > read_time:
                    read_time = j['fcounters'][f'{mod}_F_MAX_READ_TIME']
                    max_read = j['counters'][f'{mod}_MAX_READ_TIME_SIZE']

                if j['fcounters'][f'{mod}_F_MAX_WRITE_TIME'] > read_time:
                    write_time = j['fcounters'][f'{mod}_F_MAX_WRITE_TIME']
                    max_write = j['counters'][f'{mod}_MAX_WRITE_TIME_SIZE']
        return [
            (read_time, max_read),
            (write_time, max_write)
        ]

    def _get_consecs(self) -> List[int]:
        """
        Gets consecutive read and writes
        return: [total_consec_reads, total_consec_writes]
        """
        consec_read = []
        consec_write = []
        for i in self.dar_dict['POSIX']:
            consec_read.append(i['counters']['POSIX_CONSEC_READS'])
            consec_write.append(i['counters']['POSIX_CONSEC_WRITES'])
        return [sum(consec_read), sum(consec_write)]

    def parse(self, file_: str) -> Dict[str, float]:
        """
        Parses an inputted Darshan File and returns the relavent variables for Luxio
        return: darshan_variables
        """
        self.report = darshan.DarshanReport(file_, read_all=True)
        self.dar_dict = self.report.records_as_dict()
        modules_list = list(self.dar_dict.keys())
        modules_list.remove('DXT_MPIIO')
        modules_list.remove('H5F')
        total = []
        for i in modules_list:
            total.append(self._get_total_module_stats(i))

        for i, j in zip(self._sum_counters, map(sum, zip(*total))):
            self._extracted_variables[i] = j

        self._extracted_variables['total_consec_reads'], \
            self._extracted_variables['total_consec_writes'] \
            = self._get_consecs()

        self._extracted_variables['max_byte_read'], \
            self._extracted_variables['max_byte_written'] \
            = self._get_max_byte_op()

        (self._extracted_variables['max_read_time'],
         self._extracted_variables['max_read_time_size']), \
            (self._extracted_variables['max_write_time'],
             self._extracted_variables['max_write_time_size']) \
            = self._get_max_op_time_size()

        return self._output_formatter()

    def _finalize(self) -> None:
        pass
