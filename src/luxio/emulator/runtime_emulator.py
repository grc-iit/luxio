from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
from typing import Tuple

groupings  = {
    "SMALL_READ" : [
        "TOTAL_POSIX_SIZE_READ_0_100", "TOTAL_POSIX_SIZE_READ_100_1K",
        "TOTAL_MPIIO_SIZE_READ_AGG_0_100", "TOTAL_MPIIO_SIZE_READ_AGG_100_1K"
    ],
    "MEDIUM_READ" : [
        "TOTAL_POSIX_SIZE_READ_1K_10K", "TOTAL_POSIX_SIZE_READ_10K_100K", "TOTAL_POSIX_SIZE_READ_100K_1M",
        "TOTAL_MPIIO_SIZE_READ_AGG_1K_10K", "TOTAL_MPIIO_SIZE_READ_AGG_10K_100K", "TOTAL_MPIIO_SIZE_READ_AGG_100K_1M"
    ],
    "LARGE_READ" : [
        "TOTAL_POSIX_SIZE_READ_1M_4M", "TOTAL_POSIX_SIZE_READ_4M_10M", "TOTAL_POSIX_SIZE_READ_10M_100M", "TOTAL_POSIX_SIZE_READ_100M_1G", "TOTAL_POSIX_SIZE_READ_1G_PLUS",
        "TOTAL_MPIIO_SIZE_READ_AGG_1M_4M", "TOTAL_MPIIO_SIZE_READ_AGG_4M_10M", "TOTAL_MPIIO_SIZE_READ_AGG_10M_100M", "TOTAL_MPIIO_SIZE_READ_AGG_100M_1G", "TOTAL_MPIIO_SIZE_READ_AGG_1G_PLUS"
    ],
    "SMALL_WRITE" : [
        "TOTAL_POSIX_SIZE_WRITE_0_100", "TOTAL_POSIX_SIZE_WRITE_100_1K",
        "TOTAL_MPIIO_SIZE_WRITE_AGG_0_100", "TOTAL_MPIIO_SIZE_WRITE_AGG_100_1K"
    ],
    "MEDIUM_WRITE" : [
        "TOTAL_POSIX_SIZE_WRITE_1K_10K", "TOTAL_POSIX_SIZE_WRITE_10K_100K", "TOTAL_POSIX_SIZE_WRITE_100K_1M",
        "TOTAL_MPIIO_SIZE_WRITE_AGG_1K_10K", "TOTAL_MPIIO_SIZE_WRITE_AGG_10K_100K", "TOTAL_MPIIO_SIZE_WRITE_AGG_100K_1M"
    ],
    "LARGE_WRITE" : [
        "TOTAL_POSIX_SIZE_WRITE_1M_4M", "TOTAL_POSIX_SIZE_WRITE_4M_10M", "TOTAL_POSIX_SIZE_WRITE_10M_100M", "TOTAL_POSIX_SIZE_WRITE_100M_1G", "TOTAL_POSIX_SIZE_WRITE_1G_PLUS",
        "TOTAL_MPIIO_SIZE_WRITE_AGG_1M_4M", "TOTAL_MPIIO_SIZE_WRITE_AGG_4M_10M", "TOTAL_MPIIO_SIZE_WRITE_AGG_10M_100M", "TOTAL_MPIIO_SIZE_WRITE_AGG_100M_1G", "TOTAL_MPIIO_SIZE_WRITE_AGG_1G_PLUS"
    ]
}

class RuntimeEmulator():
    def __init__(self):
        self.conf = None
        return

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()
        return

    def _finalize(self):
        return

    def complex_run(self, io_traits_vec:pd.DataFrame, deployment:pd.DataFrame) -> Tuple[float,float]:
        return

    def run(self, io_traits_vec:pd.DataFrame, deployment:pd.DataFrame) -> Tuple[float,float]:
        # Estimate runtime and utilization using performance counters
        io_traits_vec = io_traits_vec.iloc[0,:]

        # Get the number of I/O and MD requests
        num_small_reads = io_traits_vec[groupings["SMALL_READ"]].sum()
        num_small_writes = io_traits_vec[groupings["SMALL_WRITE"]].sum()
        num_med_reads = io_traits_vec[groupings["MEDIUM_READ"]].sum()
        num_med_writes = io_traits_vec[groupings["MEDIUM_WRITE"]].sum()
        num_large_reads = io_traits_vec[groupings["LARGE_READ"]].sum()
        num_large_writes = io_traits_vec[groupings["LARGE_WRITE"]].sum()
        total_md_ops = io_traits_vec["TOTAL_MD_OPS"]

        #Get the amount of data for different request sizes
        small_bytes_read = num_small_reads*1*KB
        small_bytes_write = num_small_writes*1*KB
        med_bytes_read = num_med_reads*1*MB
        med_bytes_write = num_med_writes*1*MB
        large_bytes_read = num_large_reads*16*MB
        large_bytes_write = num_large_writes*16*MB
        est_total_byte_read = small_bytes_read + med_bytes_read + large_bytes_read
        est_total_byte_write = small_bytes_write + med_bytes_write + large_bytes_write
        total_byte_read = io_traits_vec["TOTAL_BYTES_READ"]
        total_byte_write = io_traits_vec["TOTAL_BYTES_WRITTEN"]

        #Scale the different byte sizes
        read_scale = total_byte_read / est_total_byte_read if est_total_byte_read > 0 else 0
        write_scale = total_byte_write / est_total_byte_write if est_total_byte_write > 0 else 0
        small_bytes_read = small_bytes_read*read_scale
        small_bytes_write = small_bytes_write*write_scale
        med_bytes_read = med_bytes_read*read_scale
        med_bytes_write = med_bytes_write*write_scale
        large_bytes_read = large_bytes_read*read_scale
        large_bytes_write = large_bytes_write*write_scale

        # Get throughputs and bandwidths for different request sizes
        write_bw_small = deployment["sequential_write_bw_small"]*MB
        read_bw_small = deployment["sequential_read_bw_small"]*MB
        write_bw_med = deployment["sequential_write_bw_medium"]*MB
        read_bw_med = deployment["sequential_read_bw_medium"]*MB
        write_bw_large = deployment["sequential_write_bw_large"]*MB
        read_bw_large = deployment["sequential_read_bw_large"]*MB
        mdm_thrpt = deployment["mdm_thrpt"]
        if mdm_thrpt == 0:
            mdm_thrpt = read_bw_small

        #Estimate I/O time
        write_time = small_bytes_write/write_bw_small + med_bytes_write/write_bw_med + large_bytes_write/write_bw_large
        read_time = small_bytes_read/read_bw_small + med_bytes_read/read_bw_med + large_bytes_read/read_bw_large
        md_time = total_md_ops / mdm_thrpt

        #Estimate runtime and disk utilization (orig times in ms)
        orig_runtime = io_traits_vec["RUNTIME"]/1000
        orig_io_time = io_traits_vec["TOTAL_IO_TIME"]/1000
        compute_time = orig_runtime - orig_io_time
        io_time = write_time + read_time + md_time
        runtime = compute_time + io_time
        utilization = io_time / runtime

        """
        print(f"total_md_ops={total_md_ops}, mdm_thrpt={mdm_thrpt}, md_time={md_time}")
        print(f"MB: large={large_bytes_write/MB} med={med_bytes_write/MB} small={small_bytes_write/MB}")
        print(f"BW: large={write_bw_large/MB} med={write_bw_med/MB} small={write_bw_small/MB}")
        print(f"orig_runtime={orig_runtime} orig_io_time={orig_io_time} compute_time={compute_time}")
        print(f"total_byte_read={total_byte_read}, total_byte_write={total_byte_write}")
        print(f"small_bytes_read={small_bytes_read}, small_bytes_write={small_bytes_write}, med_bytes_read={med_bytes_read}")
        print(f"small_bytes_write={small_bytes_write}, med_bytes_write={med_bytes_write}, large_bytes_write={large_bytes_write}")
        print(f"write_time={write_time}, read_time={read_time}, md_time={md_time}")
        print(f"io_time={io_time}")
        print()
        #"""

        return runtime,utilization
