#!/usr/bin/env python3

import json
import os
import sys
import argparse
from flux.job import Jobspec, JobWrapper
import darshan
import redis

# IOCType classes
class IOCType:
    """A type of I/O characteristic"""

    def __init__(self):
        return None


class IOCCounts(IOCType):
    def __init__(self, total_reads, total_writes, count_files):
        self.total_reads = total_reads
        self.total_writes = total_writes
        self.count_files = count_files
        return None

    def __str__(self):
        return (
            "IOCCounts"
            + "\n"
            + ("total-reads" + ": " + str(self.total_reads) + "\n")
            + ("total-writes" + ": " + str(self.total_writes) + "\n")
            + ("count-files" + ": " + str(self.count_files) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "IOCCounts"
            + " "
            + ("total-reads" + ": " + str(self.total_reads) + " ")
            + ("total-writes" + ": " + str(self.total_writes) + " ")
            + ("count-files" + ": " + str(self.count_files) + " ")
            + ">"
        )


class IOCSizes(IOCType):
    def __init__(self, total_bytes_read, total_bytes_written):
        self.total_bytes_read = total_bytes_read
        self.total_bytes_written = total_bytes_written
        return None

    def __str__(self):
        return (
            "IOCSizes"
            + "\n"
            + ("total-bytes-read" + ": " + str(self.total_bytes_read) + "\n")
            + ("total-bytes-written" + ": " + str(self.total_bytes_written) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "IOCSizes"
            + " "
            + ("total-bytes-read" + ": " + str(self.total_bytes_read) + " ")
            + ("total-bytes-written" + ": " + str(self.total_bytes_written) + " ")
            + ">"
        )


class IOCTimes(IOCType):
    def __init__(self, total_read_time_est, total_write_time_est):
        self.total_read_time_est = total_read_time_est
        self.total_write_time_est = total_write_time_est
        return None

    def __str__(self):
        return (
            "IOCTimes"
            + "\n"
            + ("total-read-time-est" + ": " + str(self.total_read_time_est) + "\n")
            + ("total-write-time-est" + ": " + str(self.total_write_time_est) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "IOCTimes"
            + " "
            + ("total-read-time-est" + ": " + str(self.total_read_time_est) + " ")
            + ("total-write-time-est" + ": " + str(self.total_write_time_est) + " ")
            + ">"
        )


# Characteristics of I/O
class IOCharacteristics:
    """Information about what makes up I/O in an application"""

    def __init__(
        self,
        total_bytes_read=0,
        total_bytes_written=0,
        total_reads=0,
        total_writes=0,
        total_read_time_est=0,
        total_write_time_est=0,
        count_files=0,
    ):
        self.characteristics = dict(
            tuple("Counts", IOCCounts(total_reads, total_writes, count_files)),
            tuple("Sizes", IOCSizes(total_bytes_read, total_bytes_written)),
            tuple("Times", IOCTimes(total_read_time_est, total_write_time_est)),
        )
        return None

    def __init__(self, json_obj=None, darshan_file=None):
        if not json_obj is None:
            fil = open(json_obj, "r")
            data = fil.read()
            obj = json.loads(data)
            self.characteristics = dict(
                [
                    tuple(
                        [
                            "Counts",
                            IOCCounts(
                                obj["total_reads"],
                                obj["total_writes"],
                                obj["count_files"],
                            ),
                        ]
                    ),
                    tuple(
                        [
                            "Sizes",
                            IOCSizes(
                                obj["total_bytes_read"], obj["total_bytes_written"]
                            ),
                        ]
                    ),
                    tuple(
                        [
                            "Times",
                            IOCTimes(
                                obj["total_read_time_est"], obj["total_write_time_est"]
                            ),
                        ]
                    ),
                ]
            )
        else:
            if not darshan_file is None:
                report = darshan.DarshanReport(darshan_file, read_all=True)
                cntrs = report.data["counters"]
                recs = report.data["records"]
                modules = recs.keys()
                total_reads = 0
                total_writes = 0
                count_files = 0
                total_bytes_read = 0
                total_bytes_written = 0
                total_read_time_est = 0
                total_write_time_est = 0
                for m in modules:
                    if m == "MPI-IO":
                        total_reads = (
                            total_reads
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_COLL_READS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_INDEP_READS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_SPLIT_READS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_NB_READS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                        )
                        total_writes = (
                            total_writes
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_COLL_WRITES")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_INDEP_WRITES")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_SPLIT_WRITES")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_NB_WRITES")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                        )
                        count_files = (
                            count_files
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_COLL_OPENS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index("MPIIO_INDEP_OPENS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                        )
                        total_bytes_read = total_bytes_read + sum(
                            [
                                rec["counters"][
                                    cntrs[m]["counters"].index("MPIIO_BYTES_READ")
                                ]
                                for rec in recs[m]
                            ]
                        )
                        total_bytes_written = total_bytes_written + sum(
                            [
                                rec["counters"][
                                    cntrs[m]["counters"].index("MPIIO_BYTES_WRITTEN")
                                ]
                                for rec in recs[m]
                            ]
                        )
                        total_read_time_est = total_read_time_est + sum(
                            [
                                rec["fcounters"][
                                    cntrs[m]["fcounters"].index("MPIIO_F_READ_TIME")
                                ]
                                for rec in recs[m]
                            ]
                        )
                        total_write_time_est = total_write_time_est + sum(
                            [
                                rec["fcounters"][
                                    cntrs[m]["fcounters"].index("MPIIO_F_WRITE_TIME")
                                ]
                                for rec in recs[m]
                            ]
                        )
                    else:
                        if True:
                            total_reads = total_reads + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index(m + "_READS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            total_writes = total_writes + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index(m + "_WRITES")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            count_files = count_files + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index(m + "_OPENS")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            total_bytes_read = total_bytes_read + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index(m + "_BYTES_READ")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            total_bytes_written = total_bytes_written + sum(
                                [
                                    rec["counters"][
                                        cntrs[m]["counters"].index(m + "_BYTES_WRITTEN")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            total_read_time_est = total_read_time_est + sum(
                                [
                                    rec["fcounters"][
                                        cntrs[m]["fcounters"].index(m + "_F_READ_TIME")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                            total_write_time_est = total_write_time_est + sum(
                                [
                                    rec["fcounters"][
                                        cntrs[m]["fcounters"].index(m + "_F_WRITE_TIME")
                                    ]
                                    for rec in recs[m]
                                ]
                            )
                self.characteristics = dict(
                    [
                        tuple(
                            [
                                "Counts",
                                IOCCounts(total_reads, total_writes, count_files),
                            ]
                        ),
                        tuple(
                            ["Sizes", IOCSizes(total_bytes_read, total_bytes_written)]
                        ),
                        tuple(
                            [
                                "Times",
                                IOCTimes(total_read_time_est, total_write_time_est),
                            ]
                        ),
                    ]
                )
        return None

    def __str__(self):
        return (
            "IOCharacteristics"
            + "\n"
            + ("characteristics" + ": " + str(self.characteristics) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "IOCharacteristics"
            + " "
            + ("characteristics" + ": " + str(self.characteristics) + " ")
            + ">"
        )


# Job Spec Internal Required Information
class JobInfo:
    def __init__(self, filename):
        self.js = Jobspec.from_yaml_file(filename)
        self.uid = os.environ["USER"]
        self.exeids = [jst["command"] for jst in self.js.tasks]
        self.runtime = self.js.duration

        def helper():
            for jst in self.js.tasks:
                resource_count = 0
                for jsr in self.js.resources:
                    if jsr["label"] == jst["slot"]:
                        if isinstance(jsr["count"], int):
                            resource_count = int(jsr["count"])
                        else:
                            resource_count = int(jsr["count"]["min"])
                        break
                yield int(jst["count"]["per_slot"]) * resource_count

        self.nprocs = [].__class__(helper())
        return None

    def __str__(self):
        return (
            "JobInfo"
            + "\n"
            + ("uid" + ": " + str(self.uid) + "\n")
            + ("exeids" + ": " + str(self.exeids) + "\n")
            + ("runtime" + ": " + str(self.runtime) + "\n")
            + ("nprocs" + ": " + str(self.nprocs) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "JobInfo"
            + " "
            + ("uid" + ": " + str(self.uid) + " ")
            + ("exeids" + ": " + str(self.exeids) + " ")
            + ("runtime" + ": " + str(self.runtime) + " ")
            + ("nprocs" + ": " + str(self.nprocs) + " ")
            + ">"
        )


# Requirements of Storage System
class StorageRequirements:
    """Requirements of storage from Characteristics"""

    def __init__(self, capacity_bytes=0, io_bandwidth_bps=0, iops=0, io_bandwidth_bps_rt=0, iops_rt=0, read_bandwidth_bps=0, write_bandwidth_bps=0, rops=0, wops=0, read_bandwidth_bps_rt=0, write_bandwidth_bps_rt=0, rops_rt=0, wops_rt=0, count_units=0):
        self.capacity_bytes = capacity_bytes
        self.io_bandwidth_bps = io_bandwidth_bps
        self.iops = iops
        self.io_bandwidth_bps_rt = io_bandwidth_bps_rt
        self.iops_rt = iops_rt
        self.read_bandwidth_bps = read_bandwidth_bps
        self.write_bandwidth_bps = write_bandwidth_bps
        self.rops = rops
        self.wops = wops
        self.read_bandwidth_bps_rt = read_bandwidth_bps_rt
        self.write_bandwidth_bps_rt = write_bandwidth_bps_rt
        self.rops_rt = rops_rt
        self.wops_rt = wops_rt
        self.count_units = count_units
        return None

    def __init__(self, job_info, io_characteristics=None, db=None):
        if io_characteristics is None:
            if db is None:
                raise Exception("No Characteristics or Database information")
            obj = db.Poll(job_info.exeids)
            if obj is None:
                self.count_units = -1
                self.io_bandwidth_bps = -1
                self.iops = -1
                self.count_units = sum(job_info.nprocs)
            self.count_units = obj["count_units"]
            self.capacity_bytes = obj["capacity_bytes"]
            self.io_bandwidth_bps = obj["io_bandwidth_bps"]
            self.io_bandwidth_bps_rt = obj["io_bandwidth_bps_rt"]
            self.read_bandwidth_bps = obj["read_bandwidth_bps"]
            self.write_bandwidth_bps = obj["write_bandwidth_bps"]
            self.read_bandwidth_bps_rt = obj["read_bandwidth_bps_rt"]
            self.write_bandwidth_bps_rt = obj["write_bandwidth_bps_rt"]
            self.iops = obj["iops"]
            self.iops_rt = obj["iops_rt"]
            self.rops = obj["rops"]
            self.wops = obj["wops"]
            self.iops_rt = obj["iops_rt"]
            self.rops_rt = obj["rops_rt"]
            self.wops_rt = obj["wops_rt"]
        else:
            self.count_units = sum(job_info.nprocs)
            self.capacity_bytes = io_characteristics.characteristics[
                "Sizes"
            ].total_bytes_written
            total_transfer_time_est = (
                io_characteristics.characteristics["Times"].total_read_time_est
                + io_characteristics.characteristics["Times"].total_write_time_est
            )
            self.io_bandwidth_bps = 0
            self.iops = 0
            self.read_bandwidth_bps = 0
            self.rops = 0
            self.write_bandwidth_bps = 0
            self.wops = 0
            self.io_bandwidth_bps_rt = 0
            self.iops_rt = 0
            self.read_bandwidth_bps_rt = 0
            self.rops_rt = 0
            self.write_bandwidth_bps_rt = 0
            self.wops_rt = 0
            if not io_characteristics.characteristics["Times"].total_read_time_est == 0:
                self.read_bandwidth_bps = (
                    io_characteristics.characteristics["Sizes"].total_bytes_read
                ) / io_characteristics.characteristics["Times"].total_read_time_est
                self.rops = (
                    io_characteristics.characteristics["Counts"].total_reads
                ) / io_characteristics.characteristics["Times"].total_read_time_est
            if not io_characteristics.characteristics["Times"].total_write_time_est == 0:
                self.write_bandwidth_bps = (
                    io_characteristics.characteristics["Sizes"].total_bytes_written
                ) / io_characteristics.characteristics["Times"].total_write_time_est
                self.wops = (
                    io_characteristics.characteristics["Counts"].total_writes
                ) / io_characteristics.characteristics["Times"].total_write_time_est
            if not total_transfer_time_est == 0:
                self.io_bandwidth_bps = (
                    io_characteristics.characteristics["Sizes"].total_bytes_read
                    + io_characteristics.characteristics[
                        "Sizes"
                    ].total_bytes_written
                ) / total_transfer_time_est
                self.iops = (
                    io_characteristics.characteristics["Counts"].total_reads
                    + io_characteristics.characteristics["Counts"].total_writes
                ) / total_transfer_time_est
            if not job_info.runtime == 0:
                self.read_bandwidth_bps_rt = (
                    io_characteristics.characteristics["Sizes"].total_bytes_read
                ) / job_info.runtime
                self.rops_rt = (
                    io_characteristics.characteristics["Counts"].total_reads
                ) / job_info.runtime
                self.write_bandwidth_bps_rt = (
                    io_characteristics.characteristics["Sizes"].total_bytes_written
                ) / job_info.runtime
                self.wops_rt = (
                    io_characteristics.characteristics["Counts"].total_writes
                ) / job_info.runtime
                self.io_bandwidth_bps_rt = (
                    io_characteristics.characteristics["Sizes"].total_bytes_read
                    + io_characteristics.characteristics[
                        "Sizes"
                    ].total_bytes_written
                ) / job_info.runtime
                self.iops_rt = (
                    io_characteristics.characteristics["Counts"].total_reads
                    + io_characteristics.characteristics["Counts"].total_writes
                ) / job_info.runtime

        return None

    def dictify(self):
        return vars(self)
    # {
    #     "capacity_bytes": int(self.capacity_bytes),
    #     "io_bandwidth_bps": int(self.io_bandwidth_bps),
    #     "iops": int(self.iops),
    #     "io_bandwidth_bps_rt": int(self.io_bandwidth_bps_rt),
    #     "iops_rt": int(self.iops_rt),
    #     "count_units": int(self.count_units),
    # }

    def __str__(self):
        return (
            "StorageRequirements"
            + "\n"
            + ("capacity-bytes" + ": " + str(self.capacity_bytes) + "\n")
            + ("io-bandwidth-bps" + ": " + str(self.io_bandwidth_bps) + "\n")
            + ("iops" + ": " + str(self.iops) + "\n")
            + ("read-bandwidth-bps" + ": " + str(self.read_bandwidth_bps) + "\n")
            + ("write-bandwidth-bps" + ": " + str(self.write_bandwidth_bps) + "\n")
            + ("rops" + ": " + str(self.rops) + "\n")
            + ("wops" + ": " + str(self.wops) + "\n")
            + ("io-bandwidth-bps-rt" + ": " + str(self.io_bandwidth_bps_rt) + "\n")
            + ("iops-rt" + ": " + str(self.iops_rt) + "\n")
            + ("read-bandwidth-bps-rt" + ": " + str(self.read_bandwidth_bps_rt) + "\n")
            + (
                "write-bandwidth-bps-rt"
                + ": "
                + str(self.write_bandwidth_bps_rt)
                + "\n"
            )
            + ("rops-rt" + ": " + str(self.rops_rt) + "\n")
            + ("wops-rt" + ": " + str(self.wops_rt) + "\n")
        )

    def __repr__(self):
        return (
            "< "
            + "StorageRequirements"
            + " "
            + ("capacity-bytes" + ": " + str(self.capacity_bytes) + " ")
            + ("io-bandwidth-bps" + ": " + str(self.io_bandwidth_bps) + " ")
            + ("iops" + ": " + str(self.iops) + " ")
            + ("read-bandwidth-bps" + ": " + str(self.read_bandwidth_bps) + " ")
            + ("write-bandwidth-bps" + ": " + str(self.write_bandwidth_bps) + " ")
            + ("rops" + ": " + str(self.rops) + " ")
            + ("wops" + ": " + str(self.wops) + " ")
            + ("io-bandwidth-bps-rt" + ": " + str(self.io_bandwidth_bps_rt) + " ")
            + ("iops-rt" + ": " + str(self.iops_rt) + " ")
            + ("read-bandwidth-bps-rt" + ": " + str(self.read_bandwidth_bps_rt) + " ")
            + ("write-bandwidth-bps-rt" + ": " + str(self.write_bandwidth_bps_rt) + " ")
            + ("rops-rt" + ": " + str(self.rops_rt) + " ")
            + ("wops-rt" + ": " + str(self.wops_rt) + " ")
            + ">"
        )

# Database class
class StorageDatabase:
    """Interface to map server software (e.g., Redis, Memcached,
etc.). Allows access to database mapping identifiers to previously
requested StorageRequirement objects"""

    def __init__(self, host="localhost", port=6379, db=0):
        self.database = redis.Redis(host=host, port=port, db=db)
        return None

    def Store(self, myid, storagereq):
        return self.database.set(json.dumps(myid), json.dumps(storagereq.dictify()))

    def Poll(self, myid):
        return json.loads(self.database.get(json.dumps(myid)))


# Front-end
if __name__ == "__main__":
    import sys

    def main(*_):
        parser = argparse.ArgumentParser()
        parser.add_argument("-j", help="Job spec (yaml,\nrequired)")
        parser.add_argument("-t", default=None, help="Trace\nfile (Darshan, optional)")
        parser.add_argument(
            "-c", default=None, help="I/O Characteristics\n(JSON, optional)"
        )
        parser.add_argument(
            "-r",
            "--enable-redis",
            help="""Enable use of
Redis to store and retrieve past requirements""",
            action="store_true",
        )
        parser.add_argument(
            "--redis-host", default="localhost", help="Host to use for redis"
        )
        parser.add_argument("--redis-port", default=6379, help="Port to use for redis")
        parser.add_argument(
            "--redis-db", default=0, help="Database number to use for redis"
        )
        parser.add_argument(
            "-v",
            "--verbose",
            help="Toggle extra detail in\noutput",
            action="store_true",
        )
        args = parser.parse_args()
        if args.t is None and args.c is None and not args.enable_redis:
            raise Exception(
                """Need some I/O Characteristic input or access to
  historical database"""
            )
        ji = JobInfo(args.j)
        ioc = (
            IOCharacteristics(json_obj=args.c)
            if not args.c is None
            else IOCharacteristics(darshan_file=args.t)
            if not args.t is None
            else None
            if args.t is None and args.c is None
            else None
        )
        db = (
            StorageDatabase(
                host=args.redis_host, port=args.redis_port, db=args.redis_db
            )
            if args.enable_redis
            else None
        )
        sr = StorageRequirements(job_info=ji, io_characteristics=ioc, db=db)
        db.Store(ji.exeids, sr) if not db is None else None
        if args.verbose:
            print(ji)
            print(ioc)
        return print(sr)

    main(*sys.argv)
