from luxio.io_requirement_extractor.trace_parser import TraceParser
import pandas as pd

class DarshanTraceParser(TraceParser):
    def standardize(self):
        df = self.df

        #Get the interface:
        df.loc[:, "INTERFACE"] = 0
        df = df.astype({"INTERFACE": int})
        df.loc[(df.TOTAL_POSIX_F_READ_TIME + df.TOTAL_POSIX_F_WRITE_TIME + df.TOTAL_POSIX_F_META_TIME) > 0, "INTERFACE"] |= 1 #POSIX
        df.loc[(df.TOTAL_MPIIO_F_READ_TIME + df.TOTAL_MPIIO_F_WRITE_TIME + df.TOTAL_MPIIO_F_META_TIME) > 0, "INTERFACE"] |= 2 #MPI
        df.loc[(df.TOTAL_STDIO_READS + df.TOTAL_STDIO_WRITES + df.TOTAL_STDIO_OPENS) > 0, "INTERFACE"] |= 4 #STDIO

        #Get total amount of I/O (MB)
        df.loc[:, "TOTAL_IO"] = (df["TOTAL_BYTES_READ"] + df["TOTAL_BYTES_WRITTEN"])/(2**20)
        df.loc[:, "TOTAL_IO_PER_PROC"] = df["TOTAL_IO"]/df.NPROCS

        #Get total number of I/O ops
        df.loc[:, "TOTAL_READ_OPS"] = (
            df.TOTAL_POSIX_READS +
            df.TOTAL_STDIO_READS +
            df.TOTAL_MPIIO_INDEP_READS +
            df.TOTAL_MPIIO_COLL_READS +
            df.TOTAL_MPIIO_SPLIT_READS +
            df.TOTAL_MPIIO_NB_READS
        )
        df.loc[:, "TOTAL_WRITE_OPS"] = (
            df.TOTAL_POSIX_WRITES +
            df.TOTAL_STDIO_WRITES +
            df.TOTAL_MPIIO_INDEP_WRITES +
            df.TOTAL_MPIIO_COLL_WRITES +
            df.TOTAL_MPIIO_SPLIT_WRITES +
            df.TOTAL_MPIIO_NB_WRITES
        )
        df.loc[:, "TOTAL_IO_OPS"] = (df.TOTAL_READ_OPS + df.TOTAL_WRITE_OPS)

        #Total IO ops for varying sizes (fractional)
        df.loc[:, "TOTAL_SIZE_IO_0_100"] = (
            df.TOTAL_POSIX_SIZE_READ_0_100 +
            df.TOTAL_POSIX_SIZE_WRITE_0_100 +
            df.TOTAL_MPIIO_SIZE_READ_AGG_0_100 +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_0_100
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_100_1K"] = (
            df.TOTAL_POSIX_SIZE_READ_100_1K +
            df.TOTAL_POSIX_SIZE_WRITE_100_1K +
            df.TOTAL_MPIIO_SIZE_READ_AGG_100_1K +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_100_1K
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_1K_10K"] = (
            df.TOTAL_POSIX_SIZE_READ_1K_10K +
            df.TOTAL_POSIX_SIZE_WRITE_1K_10K +
            df.TOTAL_MPIIO_SIZE_READ_AGG_1K_10K +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_1K_10K
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_10K_100K"] = (
            df.TOTAL_POSIX_SIZE_READ_10K_100K +
            df.TOTAL_POSIX_SIZE_WRITE_10K_100K +
            df.TOTAL_MPIIO_SIZE_READ_AGG_10K_100K +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_10K_100K
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_100K_1M"] = (
            df.TOTAL_POSIX_SIZE_READ_100K_1M +
            df.TOTAL_POSIX_SIZE_WRITE_100K_1M +
            df.TOTAL_MPIIO_SIZE_READ_AGG_100K_1M +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_100K_1M
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_0_1M"] = (
            df.TOTAL_SIZE_IO_0_100 +
            df.TOTAL_SIZE_IO_100_1K +
            df.TOTAL_SIZE_IO_100_1K +
            df.TOTAL_SIZE_IO_10K_100K +
            df.TOTAL_SIZE_IO_100K_1M
        )
        df.loc[:, "TOTAL_SIZE_IO_1M_4M"] = (
            df.TOTAL_POSIX_SIZE_READ_1M_4M +
            df.TOTAL_POSIX_SIZE_WRITE_1M_4M +
            df.TOTAL_MPIIO_SIZE_READ_AGG_1M_4M +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_1M_4M
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_4M_10M"] = (
            df.TOTAL_POSIX_SIZE_READ_4M_10M +
            df.TOTAL_POSIX_SIZE_WRITE_4M_10M +
            df.TOTAL_MPIIO_SIZE_READ_AGG_4M_10M +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_4M_10M
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_10M_100M"] = (
            df.TOTAL_POSIX_SIZE_READ_10M_100M +
            df.TOTAL_POSIX_SIZE_WRITE_10M_100M +
            df.TOTAL_MPIIO_SIZE_READ_AGG_10M_100M +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_10M_100M
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_1M_100M"] = (
            df.TOTAL_SIZE_IO_1M_4M +
            df.TOTAL_SIZE_IO_4M_10M +
            df.TOTAL_SIZE_IO_10M_100M
        )
        df.loc[:, "TOTAL_SIZE_IO_100M_1G"] = (
            df.TOTAL_POSIX_SIZE_READ_100M_1G +
            df.TOTAL_POSIX_SIZE_WRITE_100M_1G +
            df.TOTAL_MPIIO_SIZE_READ_AGG_100M_1G +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_100M_1G
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_1G_PLUS"] = (
            df.TOTAL_POSIX_SIZE_READ_1G_PLUS +
            df.TOTAL_POSIX_SIZE_WRITE_1G_PLUS +
            df.TOTAL_MPIIO_SIZE_READ_AGG_1G_PLUS +
            df.TOTAL_MPIIO_SIZE_WRITE_AGG_1G_PLUS
        )/df.TOTAL_IO_OPS
        df.loc[:, "TOTAL_SIZE_IO_100M_PLUS"] = (
            df.TOTAL_SIZE_IO_100M_1G +
            df.TOTAL_SIZE_IO_1G_PLUS
        )

        #TOTAL NUMBER OF MEDATA OPERATIONS
        df.loc[:, "TOTAL_MD_OPS"] = (
            df.TOTAL_POSIX_OPENS +
            df.TOTAL_POSIX_READS +
            df.TOTAL_POSIX_WRITES +
            df.TOTAL_POSIX_SEEKS +
            df.TOTAL_POSIX_STATS +
            df.TOTAL_POSIX_MMAPS +
            df.TOTAL_POSIX_FSYNCS +
            df.TOTAL_POSIX_FDSYNCS +

            df.TOTAL_STDIO_OPENS +
            df.TOTAL_STDIO_READS +
            df.TOTAL_STDIO_WRITES +
            df.TOTAL_STDIO_SEEKS +

            df.TOTAL_MPIIO_SYNCS
        )

        #GET TOTAL AMOUNT OF TIME SPENT IN I/O
        df.loc[:, "TOTAL_READ_TIME"] = (
            df.TOTAL_POSIX_F_READ_TIME +
            df.TOTAL_MPIIO_F_READ_TIME
        )
        df.loc[:, "TOTAL_WRITE_TIME"] = (
            df.TOTAL_POSIX_F_WRITE_TIME +
            df.TOTAL_MPIIO_F_WRITE_TIME
        )
        df.loc[:, "TOTAL_MD_TIME"] = (
            df.TOTAL_POSIX_F_META_TIME +
            df.TOTAL_MPIIO_F_META_TIME
        )
        df.loc[:, "TOTAL_IO_TIME"] = (
            df.TOTAL_READ_TIME +
            df.TOTAL_WRITE_TIME +
            df.TOTAL_MD_TIME
        )

        self.df = df
        return df
