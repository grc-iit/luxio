import darshan
from io_requirement_extractor.trace_parser import TraceParser

class DarshanTraceParser():
    extracted_variables = {}
    sum_counters = [
        'total_reads',
        'total_writes'
        'total_bytes_read',
        'total_bytes_written',
        'read_time',
        'write_time',
    ]

    def __init__(self) -> None:
        pass

    def _getTotalModuleStats(self, module_: str):
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

    def _getConsecs(self):
        consec_read = []
        consec_write = []
        for i in self.dar_dict['POSIX']:
            consec_read.append(i['counters']['POSIX_CONSEC_READS'])
            consec_write.append(i['counters']['POSIX_CONSEC_WRITES'])
        return [sum(consec_read), sum(consec_write)]

    def parse(self, file_: str) -> None:
        self.report = darshan.DarshanReport(file_, read_all=True)
        self.dar_dict = self.report.records_as_dict()
        modules_list = list(self.dar_dict.keys())
        modules_list.remove('DXT_MPIIO')
        modules_list.remove('H5F')
        total = []
        for i in modules_list:
            total.append(self._getTotalModuleStats(i))

        for i, j in zip(self.sum_counters, map(sum, zip(*total))):
            self.extracted_variables[i] = j

        self.extracted_variables['total_consec_reads'], \
            self.extracted_variables['total_consec_writes'] \
            = self._getConsecs()

        return self.extracted_variables

    def _finalize(self) -> None:
        pass
