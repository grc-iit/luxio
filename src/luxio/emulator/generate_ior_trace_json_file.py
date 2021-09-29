from luxio.external_clients.json_client import *
from luxio.common.constants import *

class GenIORTraceJsonFile():
    def __init__(self, read_bw_small=7.68, read_bw_med=110, read_bw_large=112, write_bw_small=7.68, write_bw_med=110, write_bw_large=112, mdm_thrpt=0):
        self.read_bw_small = read_bw_small*MB
        self.read_bw_med = read_bw_med*MB
        self.read_bw_large = read_bw_large*MB
        self.write_bw_small = write_bw_small*MB
        self.write_bw_med = write_bw_med*MB
        self.write_bw_large = write_bw_large*MB
        self.mdm_thrpt = mdm_thrpt
        if mdm_thrpt == 0:
            self.mdm_thrpt = self.read_bw_small
        self.ior_trace_dict = {
            "RUNTIME": 0,
            "TOTAL_READ_TIME": 0,
            "TOTAL_WRITE_TIME": 0,
            "NPROCS": 0,
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

    def fixed_compute(self, num_checkpoints, sleep_size, io_intensity, rfrac, req_size, num_procs, file_per_proc, fsync_per_write, sync, randomness):
        """
        Calculate # of requests to get runtime given compute
        req_size is in bytes
        runtime is in seconds
        io_intensity is between 0 and 1
        """
        compute_time = sleep_size * num_checkpoints
        io_time = compute_time/(1-io_intensity) - compute_time
        read_time = io_time * rfrac
        write_time = io_time * (1-rfrac)
        if req_size <= 1*KB:
            num_reads = int(read_time * self.read_bw_small / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_small / req_size / num_checkpoints)
        elif req_size <= 1*MB:
            num_reads = int(read_time * self.read_bw_med / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_med / req_size / num_checkpoints)
        else:
            num_reads = int(read_time * self.read_bw_large / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_large / req_size / num_checkpoints)
        self.num_reads = num_reads
        self.num_writes = num_writes
        return self.gen(num_checkpoints, sleep_size, num_procs, num_reads, num_writes, req_size, file_per_proc, fsync_per_write, sync, randomness)

    def cap_runtime(self, num_checkpoints, io_intensity, rfrac, req_size, runtime, num_procs, file_per_proc, fsync_per_write, sync, randomness):
        """
        Calculate # of requests to get runtime given compute
        req_size is in bytes
        runtime is in seconds
        io_intensity is between 0 and 1
        """
        io_time = runtime*io_intensity
        compute_time = runtime - io_time
        sleep_size = compute_time / num_checkpoints
        read_time = io_time * rfrac
        write_time = io_time * (1-rfrac)
        if req_size <= 1*KB:
            num_reads = int(read_time * self.read_bw_small / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_small / req_size / num_checkpoints)
        elif req_size <= 1*MB:
            num_reads = int(read_time * self.read_bw_med / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_med / req_size / num_checkpoints)
        else:
            num_reads = int(read_time * self.read_bw_large / req_size / num_checkpoints)
            num_writes = int(write_time * self.write_bw_large / req_size / num_checkpoints)
        self.num_reads = num_reads
        self.num_writes = num_writes
        return self.gen(num_checkpoints, sleep_size, num_procs, num_reads, num_writes, req_size, file_per_proc, fsync_per_write, sync, randomness)

    def gen(self, num_checkpoints, sleep_size, num_procs, num_reads, num_writes, req_size, file_per_proc, fsync_per_write, fsync, randomness) -> dict:
        """
        sleep_size in seconds
        num_reads/writes per checkpoint
        req_size in bytes
        """

        self.ior_trace_dict['NPROCS'] = num_procs
        for i in range(num_checkpoints):
            self.ior_trace_dict['TOTAL_BYTES_READ'] += num_reads * req_size
            self.ior_trace_dict['TOTAL_BYTES_WRITTEN'] += num_writes * req_size
            self.ior_trace_dict['TOTAL_POSIX_MAX_BYTE_READ'] = req_size
            self.ior_trace_dict['TOTAL_POSIX_MAX_BYTE_WRITTEN'] = req_size

            if randomness == 1:
                # random read and write
                self.ior_trace_dict['TOTAL_POSIX_CONSEC_READS'] += 0
                self.ior_trace_dict['TOTAL_POSIX_CONSEC_WRITES'] += 0
                self.ior_trace_dict['TOTAL_POSIX_SEQ_READS'] += 0.6 * num_reads
                self.ior_trace_dict['TOTAL_POSIX_SEQ_WRITES'] += 0.7 * num_writes
                self.ior_trace_dict['TOTAL_POSIX_SEEKS'] += num_reads + num_writes
            else:
                self.ior_trace_dict['TOTAL_POSIX_CONSEC_READS'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_CONSEC_WRITES'] += num_writes
                self.ior_trace_dict['TOTAL_POSIX_SEQ_READS'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SEQ_WRITES'] += num_writes
                self.ior_trace_dict['TOTAL_POSIX_SEEKS'] += 0

            if file_per_proc == 1:
                self.ior_trace_dict['TOTAL_POSIX_OPENS'] += num_procs
                if fsync == 1:
                    if fsync_per_write == 1:
                        self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] += num_writes
                    else:
                        self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] += num_procs
            else:
                self.ior_trace_dict['TOTAL_POSIX_OPENS'] += 1
                if fsync == 1:
                    if fsync_per_write == 1:
                        self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] += num_writes
                    else:
                        self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] += 1

            self.ior_trace_dict['TOTAL_POSIX_READS'] += num_reads
            self.ior_trace_dict['TOTAL_POSIX_WRITES'] += num_writes

            if req_size >= 0 and req_size <= 100:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_0_100'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_0_100'] += num_writes
            elif req_size > 100 and req_size <= 1*KB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100_1K'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100_1K'] += num_writes
            elif req_size > 1*KB and req_size <= 10*KB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1K_10K'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1K_10K'] += num_writes
            elif req_size > 10*KB and req_size <= 100*KB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10K_100K'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10K_100K'] += num_writes
            elif req_size > 100*KB and req_size <= 1*MB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100K_1M'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100K_1M'] += num_writes
            elif req_size > 1*MB and req_size <= 4*MB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1M_4M'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1M_4M'] += num_writes
            elif req_size > 4*MB and req_size <= 10*MB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_4M_10M'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_4M_10M'] += num_writes
            elif req_size > 10*MB and req_size <= 100*MB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10M_100M'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10M_100M'] += num_writes
            elif req_size > 100*MB and req_size <= 1*GB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100M_1G'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100M_1G'] += num_writes
            elif req_size > 1*GB:
                self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1G_PLUS'] += num_reads
                self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1G_PLUS'] += num_writes

            # others
            self.ior_trace_dict['INTERFACE'] = 1
            self.ior_trace_dict['TOTAL_IO'] += (self.ior_trace_dict['TOTAL_BYTES_READ'] + self.ior_trace_dict['TOTAL_BYTES_WRITTEN'])/(2**20) # unit: MB
            self.ior_trace_dict['TOTAL_IO_PER_PROC'] += self.ior_trace_dict['TOTAL_IO']/self.ior_trace_dict['NPROCS']
            self.ior_trace_dict['TOTAL_READ_OPS'] += self.ior_trace_dict['TOTAL_POSIX_READS'] + self.ior_trace_dict['TOTAL_STDIO_READS']
            self.ior_trace_dict['TOTAL_WRITE_OPS'] += self.ior_trace_dict['TOTAL_POSIX_WRITES'] + self.ior_trace_dict['TOTAL_STDIO_WRITES']
            self.ior_trace_dict['TOTAL_IO_OPS'] += self.ior_trace_dict['TOTAL_READ_OPS'] + self.ior_trace_dict['TOTAL_WRITE_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_0_100'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_0_100'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_0_100'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_100_1K'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100_1K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100_1K'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_1K_10K'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1K_10K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1K_10K'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_10K_100K'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10K_100K'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10K_100K'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_100K_1M'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100K_1M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100K_1M'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_0_1M'] += self.ior_trace_dict['TOTAL_SIZE_IO_0_100'] + self.ior_trace_dict['TOTAL_SIZE_IO_100_1K'] + \
                                                        self.ior_trace_dict['TOTAL_SIZE_IO_1K_10K'] + self.ior_trace_dict['TOTAL_SIZE_IO_10K_100K'] + \
                                                        self.ior_trace_dict['TOTAL_SIZE_IO_100K_1M']
            self.ior_trace_dict['TOTAL_SIZE_IO_1M_4M'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1M_4M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1M_4M'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_4M_10M'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_4M_10M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_4M_10M'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_10M_100M'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_10M_100M'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_10M_100M'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_1M_100M'] += self.ior_trace_dict['TOTAL_SIZE_IO_1M_4M'] + self.ior_trace_dict['TOTAL_SIZE_IO_4M_10M'] + \
                                                           self.ior_trace_dict['TOTAL_SIZE_IO_10M_100M']
            self.ior_trace_dict['TOTAL_SIZE_IO_100M_1G'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_100M_1G'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_100M_1G'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_1G_PLUS'] += (self.ior_trace_dict['TOTAL_POSIX_SIZE_READ_1G_PLUS'] + self.ior_trace_dict['TOTAL_POSIX_SIZE_WRITE_1G_PLUS'])/self.ior_trace_dict['TOTAL_IO_OPS']
            self.ior_trace_dict['TOTAL_SIZE_IO_100M_PLUS'] += self.ior_trace_dict['TOTAL_SIZE_IO_100M_1G'] + self.ior_trace_dict['TOTAL_SIZE_IO_1G_PLUS']
            self.ior_trace_dict['TOTAL_MD_OPS'] += self.ior_trace_dict['TOTAL_POSIX_OPENS'] + self.ior_trace_dict['TOTAL_POSIX_SEEKS'] + \
                                                  self.ior_trace_dict['TOTAL_POSIX_STATS'] + self.ior_trace_dict['TOTAL_POSIX_MMAPS'] + \
                                                  self.ior_trace_dict['TOTAL_POSIX_FDSYNCS'] + self.ior_trace_dict['TOTAL_POSIX_FSYNCS'] #+ \
                                                  #self.ior_trace_dict['TOTAL_POSIX_READS'] + self.ior_trace_dict['TOTAL_POSIX_WRITES']
        #Time in milliseconds
        if req_size <= 1*KB:
            self.ior_trace_dict['TOTAL_READ_TIME'] = 1000*num_checkpoints * (num_reads) * req_size / self.read_bw_small
            self.ior_trace_dict['TOTAL_WRITE_TIME'] = 1000*num_checkpoints * (num_writes) * req_size / self.write_bw_small
        elif req_size <= 1*MB:
            self.ior_trace_dict['TOTAL_READ_TIME'] = 1000*num_checkpoints * (num_reads) * req_size / self.read_bw_med
            self.ior_trace_dict['TOTAL_WRITE_TIME'] = 1000*num_checkpoints * (num_writes) * req_size / self.write_bw_med
        else:
            self.ior_trace_dict['TOTAL_READ_TIME'] = 1000*num_checkpoints * (num_reads) * req_size / self.read_bw_large
            self.ior_trace_dict['TOTAL_WRITE_TIME'] = 1000*num_checkpoints * (num_writes) * req_size / self.write_bw_large
        self.ior_trace_dict['TOTAL_MD_TIME'] = 1000*(self.ior_trace_dict['TOTAL_MD_OPS']) / self.mdm_thrpt

        self.ior_trace_dict['TOTAL_IO_TIME'] = (
            self.ior_trace_dict['TOTAL_READ_TIME'] +
            self.ior_trace_dict['TOTAL_WRITE_TIME'] +
            self.ior_trace_dict['TOTAL_MD_TIME']
        )

        """
        print(f"TOTAL_MD_OPS={self.ior_trace_dict['TOTAL_MD_OPS']}")
        print(f"MD_THRPT={self.mdm_thrpt}")
        print(f"TOTAL_WRITE_TIME={self.ior_trace_dict['TOTAL_WRITE_TIME']/1000}")
        print(f"TOTAL_MD_TIME={self.ior_trace_dict['TOTAL_MD_TIME']/1000}")
        print()
        """

        self.ior_trace_dict['RUNTIME'] = self.ior_trace_dict['TOTAL_IO_TIME'] + num_checkpoints*sleep_size*1000
        return self

    def gen_time_series(self, num_checkpoints, sleep_size, num_procs, num_reads, num_writes, req_size, file_per_proc, fsync_per_write, fsync, randomness):
        ts = []
        for checkpoint in range(2*num_checkpoints):
            if checkpoint % 2 == 0:
                ts.append(GenIORTraceJsonFile(
                    read_bw_small = self.read_bw_small/MB,
                    read_bw_med = self.read_bw_med/MB,
                    read_bw_large = self.read_bw_large/MB,
                    write_bw_small = self.write_bw_small/MB,
                    write_bw_med = self.write_bw_med/MB,
                    write_bw_large = self.write_bw_large/MB,
                    mdm_thrpt = self.mdm_thrpt
                ).gen(1, sleep_size, 0, 0, 0, 0, 0, 0, 0, 0))
            if checkpoint % 2 == 1:
                ts.append(GenIORTraceJsonFile(
                    read_bw_small = self.read_bw_small/MB,
                    read_bw_med = self.read_bw_med/MB,
                    read_bw_large = self.read_bw_large/MB,
                    write_bw_small = self.write_bw_small/MB,
                    write_bw_med = self.write_bw_med/MB,
                    write_bw_large = self.write_bw_large/MB,
                    mdm_thrpt = self.mdm_thrpt
                ).gen(
                    1, 0, num_procs,
                    num_reads, num_writes, req_size,
                    file_per_proc, fsync_per_write, fsync, randomness)
                )

    def get(self):
        return self.ior_trace_dict

    def to_json(self, file_name: str):
        """
        Save the ior trace as a json file
        :param ior_trace: dict
        :param filename: str
        """
        JSONClient().save(self.ior_trace_dict, file_name)
