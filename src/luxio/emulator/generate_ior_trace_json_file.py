from luxio.external_clients.json_client import *

class GenIORTraceJsonFile():
    def __init__(self):
        self.ior_trace_dict = {
            "NRPOCS": 0,
            "TOTAL_BYTES_READ": 0,
            "TOTAL_BYTES_WRITTEN": 0,
            "TOTAL_MPIIO_COLL_OPENS": 0,
            "TOTAL_MPIIO_COLL_READS": 0,
            "TOTAL_MPIIO_COLL_WRITES": 0,
            "TOTAL_MPIIO_HINTS": 0,
            "TOTAL_MPIIO_INDEP_OPENS": 0,
            "TOTAL_MPIIO_INDEP_READS": 0,
            "TOTAL_MPIIO_INDEP_WRITES": 0,
            "TOTAL_MPIIO_NB_READS": 0,
            "TOTAL_MPIIO_NB_WRITES": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_0_100": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_100K_1M": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_100M_1G": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_100_1K": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_10K_100K": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_10M_100M": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_1G_PLUS": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_1K_10K": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_1M_4M": 0,
            "TOTAL_MPIIO_SIZE_READ_AGG_4M_10M": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_0_100": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_100K_1M": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_100M_1G": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_100_1K": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_10K_100K": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_10M_100M": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_1G_PLUS": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_1K_10K": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_1M_4M": 0,
            "TOTAL_MPIIO_SIZE_WRITE_AGG_4M_10M": 0,
            "TOTAL_MPIIO_SPLIT_READS": 0,
            "TOTAL_MPIIO_SPLIT_WRITES": 0,
            "TOTAL_MPIIO_SYNCS": 0,
            "TOTAL_MPIIO_VIEWS": 0,
            "TOTAL_POSIX_CONSEC_READS": 0,
            "TOTAL_POSIX_CONSEC_WRITES": 0,
            "TOTAL_POSIX_FDSYNCS": 0,
            "TOTAL_POSIX_FILE_ALIGNMENT": 262144,
            "TOTAL_POSIX_FILE_NOT_ALIGNED": 0,
            "TOTAL_POSIX_FSYNCS": 0,
            "TOTAL_POSIX_MAX_BYTE_READ": 0,
            "TOTAL_POSIX_MAX_BYTE_WRITTEN": 0,
            "TOTAL_POSIX_MEM_ALIGNMENT": 64,
            "TOTAL_POSIX_MEM_NOT_ALIGNED": 0,
            "TOTAL_POSIX_MMAPS": 0,
            "TOTAL_POSIX_MODE": 438,
            "TOTAL_POSIX_OPENS": 0,
            "TOTAL_POSIX_READS": 0,
            "TOTAL_POSIX_RW_SWITCHES": 0,
            "TOTAL_POSIX_SEEKS": 0,
            "TOTAL_POSIX_SEQ_READS": 0,
            "TOTAL_POSIX_SEQ_WRITES": 0,
            "TOTAL_POSIX_SIZE_READ_0_100": 0,
            "TOTAL_POSIX_SIZE_READ_100K_1M": 0,
            "TOTAL_POSIX_SIZE_READ_100M_1G": 0,
            "TOTAL_POSIX_SIZE_READ_100_1K": 0,
            "TOTAL_POSIX_SIZE_READ_10K_100K": 0,
            "TOTAL_POSIX_SIZE_READ_10M_100M": 0,
            "TOTAL_POSIX_SIZE_READ_1G_PLUS": 0,
            "TOTAL_POSIX_SIZE_READ_1K_10K": 0,
            "TOTAL_POSIX_SIZE_READ_1M_4M": 0,
            "TOTAL_POSIX_SIZE_READ_4M_10M": 0,
            "TOTAL_POSIX_SIZE_WRITE_0_100": 0,
            "TOTAL_POSIX_SIZE_WRITE_100K_1M": 0,
            "TOTAL_POSIX_SIZE_WRITE_100M_1G": 0,
            "TOTAL_POSIX_SIZE_WRITE_100_1K": 0,
            "TOTAL_POSIX_SIZE_WRITE_10K_100K": 0,
            "TOTAL_POSIX_SIZE_WRITE_10M_100M": 0,
            "TOTAL_POSIX_SIZE_WRITE_1G_PLUS": 0,
            "TOTAL_POSIX_SIZE_WRITE_1K_10K": 0,
            "TOTAL_POSIX_SIZE_WRITE_1M_4M": 0,
            "TOTAL_POSIX_SIZE_WRITE_4M_10M": 0,
            "TOTAL_POSIX_STATS": 0,
            "TOTAL_POSIX_WRITES": 0,
            "TOTAL_STDIO_OPENS": 0,
            "TOTAL_STDIO_READS": 0,
            "TOTAL_STDIO_SEEKS": 0,
            "TOTAL_STDIO_WRITES": 0,
            "TOTAL_IO": 0,
            "TOTAL_IO_PER_PROC": 0,
            "TOTAL_READ_OPS": 0,
            "TOTAL_WRITE_OPS": 0,
            "TOTAL_IO_OPS": 0,
            "TOTAL_SIZE_IO_0_100": 0,
            "TOTAL_SIZE_IO_100_1K": 0,
            "TOTAL_SIZE_IO_1K_10K": 0,
            "TOTAL_SIZE_IO_10K_100K": 0,
            "TOTAL_SIZE_IO_100K_1M": 0,
            "TOTAL_SIZE_IO_0_1M": 0,
            "TOTAL_SIZE_IO_1M_4M": 0,
            "TOTAL_SIZE_IO_4M_10M": 0,
            "TOTAL_SIZE_IO_10M_100M": 0,
            "TOTAL_SIZE_IO_1M_100M": 0,
            "TOTAL_SIZE_IO_100M_1G": 0,
            "TOTAL_SIZE_IO_1G_PLUS": 0,
            "TOTAL_SIZE_IO_100M_PLUS": 0,
            "TOTAL_MD_OPS": 0
        }
        return

    def _initialize(self):
        return

    def _finalize(self):
        return

    # ior command line parameters (-e:perform fsyn upon posix write close -F: filePerProc -Y: fsyncPerWrite -z: randomness)
    # randomness=1 ===> random read/write      randomness=0 ====> sequential read/write
    # fsyn=1 ==> perform fsyn when file close  fsyn=0 ==> disable fsyn
    # file_per_proc=1 ==> each proc has a file    file_per_proc ====> only one file
    # fsync_per_write=1 ===> perform fsyn after each write    fsync_per_write=0 ===> disable fsync_per_write
    def gen_IOR_trace(self, num_procs, num_reads, num_writes, req_size, file_per_proc, fsync_per_write, fsync, randomness) -> dict:
        self.ior_trace_dict['NRPOCS'] = num_procs
        self.ior_trace_dict['TOTAL_BYTES_READ'] = num_reads * req_size
        self.ior_trace_dict['TOTAL_BYTES_WRITTEN'] = num_writes * req_size
        self.ior_trace_dict['TOTAL_POSIX_MAX_BYTE_READ'] = req_size
        self.ior_trace_dict['TOTAL_POSIX_MAX_BYTE_WRITTEN'] = req_size

        if randomness == 1:
            # random read and write
            self.ior_trace_dict['TOTAL_POSIX_CONSEC_READS'] = 0
            self.ior_trace_dict['TOTAL_POSIX_CONSEC_WRITES'] = 0
            self.ior_trace_dict['TOTAL_POSIX_SEQ_READS'] = 0.6 * num_reads
            self.ior_trace_dict['TOTAL_POSIX_SEQ_WRITES'] = 0.7 * num_writes
            self.ior_trace_dict['TOTAL_POSIX_SEEKS'] = num_reads + num_writes
        else:
            self.ior_trace_dict['TOTAL_POSIX_CONSEC_READS'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_CONSEC_WRITES'] = num_writes
            self.ior_trace_dict['TOTAL_POSIX_SEQ_READS'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SEQ_WRITES'] = num_writes
            self.ior_trace_dict['TOTAL_POSIX_SEEKS'] = 0

        if file_per_proc == 1:
            self.ior_trace_dict['TOTAL_POSIX_OPENS'] = num_procs
            if fsync == 1:
                if fsync_per_write == 1:
                    self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] = num_writes
                else:
                    self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] = num_procs
        else:
            self.ior_trace_dict['TOTAL_POSIX_OPENS'] = 1
            if fsync == 1:
                if fsync_per_write == 1:
                    self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] = num_writes
                else:
                    self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] = 1

        self.ior_trace_dict['TOTAL_POSIX_READS'] = num_reads
        self.ior_trace_dict['TOTAL_POSIX_WRITES'] = num_writes

        if req_size >= 0 and req_size < 100:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_0_100'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_0_100'] = num_writes
        elif req_size >= 100 and req_size < 1024:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100_1K'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100_1K'] = num_writes
        elif req_size >= 1024 and req_size < 10240:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1K_10K'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1K_10K'] = num_writes
        elif req_size >= 10240 and req_size < 102400:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10K_100K'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10K_100K'] = num_writes
        elif req_size >= 102400 and req_size < 1048576:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100K_1M'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100K_1M'] = num_writes
        elif req_size >= 1048576 and req_size < 4194304:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1M_4M'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1M_4M'] = num_writes
        elif req_size >= 4194304 and req_size < 10485760:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_4M_10M'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_4M_10M'] = num_writes
        elif req_size >= 10485760 and req_size < 104857600:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10M_100M'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10M_100M'] = num_writes
        elif req_size >= 104857600 and req_size < 1073741824:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100M_1G'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100M_1G'] = num_writes
        elif req_size >= 1073741824:
            self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1G_PLUS'] = num_reads
            self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1G_PLUS'] = num_writes

        # others
        self.ior_trace_dict['INTERFACE'] = 1
        self.ior_trace_dict['TOTAL_IO'] = (self.ior_trace_dict['TOTAL_BYTES_READ'] + self.ior_trace_dict['TOTAL_BYTES_WRITTEN'])/(2**20) # unit: MB
        self.ior_trace_dict['TOTAL_IO_PER_PROC'] = self.ior_trace_dict['TOTAL_IO']/self.ior_trace_dict['NRPOCS']
        self.ior_trace_dict['TOTAL_READ_OPS'] = self.ior_trace_dict['TOTAL_POSIX_READS'] + self.ior_trace_dict['TOTAL_STDIO_READS']
        self.ior_trace_dict['TOTAL_WRITE_OPS'] = self.ior_trace_dict['TOTAL_POSIX_WRITES'] + self.ior_trace_dict['TOTAL_STDIO_WRITES']
        self.ior_trace_dict['TOTAL_IO_OPS'] = self.ior_trace_dict['TOTAL_READ_OPS'] + self.ior_trace_dict['TOTAL_WRITE_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_0_100'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_0_100'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_0_100'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_100_1K'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100_1K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100_1K'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_1K_10K'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1K_10K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1K_10K'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_10K_100K'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10K_100K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10K_100K'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_100K_1M'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100K_1M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100K_1M'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_0_1M'] = self.ior_trace_dict['TOTAL_SIZE_IO_0_100'] + self.ior_trace_dict['TOTAL_SIZE_IO_100_1K'] + \
                                                    self.ior_trace_dict['TOTAL_SIZE_IO_1K_10K'] + self.ior_trace_dict['TOTAL_SIZE_IO_10K_100K'] + \
                                                    self.ior_trace_dict['TOTAL_SIZE_IO_100K_1M']
        self.ior_trace_dict['TOTAL_SIZE_IO_1M_4M'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1M_4M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1M_4M'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_4M_10M'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_4M_10M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_4M_10M'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_10M_100M'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10M_100M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10M_100M'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_1M_100M'] = self.ior_trace_dict['TOTAL_SIZE_IO_1M_4M'] + self.ior_trace_dict['TOTAL_SIZE_IO_4M_10M'] + \
                                                       self.ior_trace_dict['TOTAL_SIZE_IO_10M_100M']
        self.ior_trace_dict['TOTAL_SIZE_IO_100M_1G'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100M_1G'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100M_1G'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_1G_PLUS'] = (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1G_PLUS'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1G_PLUS'])/self.ior_trace_dict['TOTAL_IO_OPS']
        self.ior_trace_dict['TOTAL_SIZE_IO_100M_PLUS'] = self.ior_trace_dict['TOTAL_SIZE_IO_100M_1G'] + self.ior_trace_dict['TOTAL_SIZE_IO_1G_PLUS']
        self.ior_trace_dict['TOTAL_MD_OPS'] = self.ior_trace_dict['TOTAL_POSIX_OPENS'] + self.ior_trace_dict['TOTAL_POSIX_READS'] + \
                                              self.ior_trace_dict['TOTAL_POSIX_WRITES'] + self.ior_trace_dict['TOTAL_POSIX_SEEKS'] + \
                                              self.ior_trace_dict['TOTAL_POSIX_STATS'] + self.ior_trace_dict['TOTAL_POSIX_MMAPS'] + \
                                              self.ior_trace_dict['TOTAL_POSIX_FDSYNCS'] + self.ior_trace_dict['TOTAL_POSIX_FSYNCS']

        return self.ior_trace_dict

    def gen_JSON_file(self, ior_trace: dict, file_name: str):
        """
        Save the ior trace as a json file
        :param ior_trace: dict
        :param filename: str
        """
        JSONClient().save(ior_trace, file_name)
