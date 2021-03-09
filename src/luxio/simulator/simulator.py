from luxio.common.configuration_manager import *
import pandas as pd

class Simulator():
    def __init__(self):
        self.conf = None
        return

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()
        return

    def _finalize(self):
        return

    def simulate_apps_runtime(self, darshan_trace_path:str, qosas_csv_path:str):
        # Load the darshan trace
        self.apps_trace_df = pd.read_csv(darshan_trace_path)
        # Load the Qosas
        self.qosas_df = pd.read_csv(qosas_csv_path)

        # Calculate each application's runtime on each Qosa
        idx = 0
        apps_runtime_df = pd.DataFrame(columns=['APP_NAME', 'Qosa_Seq', 'IO_runtime'])
        for trace_idx in self.apps_trace_df.index:
            app_trace_df = self.apps_trace_df.loc[[trace_idx]]
            for qosa_idx in self.qosas_df.index:
                qosa_df = self.qosas_df.loc[[qosa_idx]]
                apps_runtime_df.loc[idx, 'APP_NAME'] = app_trace_df.at[trace_idx, 'APP_NAME']
                apps_runtime_df.loc[idx, 'Qosa_Seq'] = qosa_idx + 1
                apps_runtime_df.loc[idx, 'IO_runtime'] = self._estimate_app_runtime(trace_idx, app_trace_df, qosa_idx, qosa_df)
                idx += 1

        # save the result into a csv file
        apps_runtime_df.to_csv(self.conf.apps_runtime_csv_path, index=False)


    def _estimate_app_runtime(self, trace_idx: int, darshan_trace:pd.DataFrame, qosa_idx:int, qosa:pd.DataFrame) -> float:
        # Estimate the application io_runtime on the qosa (io_time = read_time + write_time + metadata_time)

        # Getting the app information from the darshan_trace. Including: TOTAL_POSIX_CONSEC_READS, TOTAL_POSIX_CONSEC_WRITES,
        # TOTAL_SIZE_IO_0_1M, TOTAL_SIZE_IO_1M_100M, TOTAL_SIZE_IO_100M_PLUS, TOTAL_BYTES_READ, TOTAL_BYTES_WRITTEN,
        # TOTAL_MD_OPS
        posix_consec_reads = darshan_trace.at[trace_idx, "TOTAL_POSIX_CONSEC_READS"]
        posix_consec_writes = darshan_trace.at[trace_idx, "TOTAL_POSIX_CONSEC_WRITES"]
        total_size_io_0_1m = darshan_trace.at[trace_idx, "TOTAL_SIZE_IO_0_1M"]
        total_size_io_1m_100m = darshan_trace.at[trace_idx, "TOTAL_SIZE_IO_1M_100M"]
        total_size_io_100m_plus = darshan_trace.at[trace_idx, "TOTAL_SIZE_IO_100M_PLUS"]
        total_byte_read = darshan_trace.at[trace_idx, "TOTAL_BYTES_READ"]
        total_byte_written = darshan_trace.at[trace_idx, "TOTAL_BYTES_WRITTEN"]
        total_md_ops = darshan_trace.at[trace_idx, "TOTAL_MD_OPS"]

        # Getting the qosa information from the QOSA dataframe. Including: sequential_mdm_thrpt_1024, sequential_write_bw_1024,
        # sequential_read_bw_1024,sequential_mdm_thrpt_16777216, sequential_write_bw_16777216,sequential_read_bw_16777216,
        # random_mdm_thrpt_1024,random_write_bw_1024, random_read_bw_1024,random_mdm_thrpt_16777216,random_write_bw_16777216,
        # random_read_bw_16777216
        seq_mdm_thrpt_1024 = qosa.at[qosa_idx, "sequential_mdm_thrpt_1024"]
        seq_write_bw_1024 = qosa.at[qosa_idx, "sequential_write_bw_1024"]
        seq_read_bw_1024 = qosa.at[qosa_idx, "sequential_read_bw_1024"]
        seq_mdm_thrpt_16m = qosa.at[qosa_idx, "sequential_mdm_thrpt_16777216"]
        seq_write_bw_16m = qosa.at[qosa_idx, "sequential_write_bw_16777216"]
        seq_read_bw_16m = qosa.at[qosa_idx, "sequential_read_bw_16777216"]
        rnd_mdm_thrpt_1024 = qosa.at[qosa_idx, "random_mdm_thrpt_1024"]
        rnd_write_bw_1024 = qosa.at[qosa_idx, "random_write_bw_1024"]
        rnd_read_bw_1024 = qosa.at[qosa_idx, "random_read_bw_1024"]
        rnd_mdm_thrpt_16m = qosa.at[qosa_idx, "random_mdm_thrpt_16777216"]
        rnd_write_bw_16m = qosa.at[qosa_idx, "random_write_bw_16777216"]
        rnd_read_bw_16m = qosa.at[qosa_idx, "random_read_bw_16777216"]

        total_size_big_io = total_size_io_1m_100m + total_size_io_100m_plus

        io_time = 0
        if posix_consec_reads > 0 or posix_consec_writes > 0:
            # sequential read/write
            if total_size_io_0_1m > total_size_big_io:
                #small read/write
                write_time = total_byte_written / seq_write_bw_1024
                read_time = total_byte_read / seq_read_bw_1024
                md_time = total_md_ops / seq_mdm_thrpt_1024
                io_time = write_time + read_time + md_time
            else:
                #large read/write
                write_time = total_byte_written / seq_write_bw_16m
                read_time = total_byte_read / seq_read_bw_16m
                md_time = total_md_ops / seq_mdm_thrpt_16m
                io_time = write_time + read_time + md_time
        else:
            # random read/write
            if total_size_io_0_1m > total_size_big_io:
                #small read/write
                write_time = total_byte_written / rnd_write_bw_1024
                read_time = total_byte_read / rnd_read_bw_1024
                md_time = total_md_ops / rnd_mdm_thrpt_1024
                io_time = write_time + read_time + md_time
            else:
                #large read/write
                write_time = total_byte_written / rnd_write_bw_16m
                read_time = total_byte_read / rnd_read_bw_16m
                md_time = total_md_ops / rnd_mdm_thrpt_16m
                io_time = write_time + read_time + md_time

        return io_time;