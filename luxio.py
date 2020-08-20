#!/usr/bin/env python3

import argparse
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from storage_requirement_builder.storage_requirement_builder import *
from storage_configurator.storage_configurator_factory import *
from external_clients.json_client import *

class LuxioBin:
    def __init__(self, args):
        self.conf = ConfigurationManager.get_instance()
        self.conf.job_spec=args.j #"sample/job_info.json"
        self.conf.darshan_trace_path=args.t #"sample/sample.darshan"
        self.conf.io_req_out_path=args.i #"sample/io_req_output.json"
        self.conf.storage_req_out_path=args.s #"sample/stor_req_output.json"
        self.conf.storage_req_config_out_path=args.c #"sample/stor_req_conf_output.json"
        self.conf.db_type = KVStoreType.REDIS
        self.conf.db_addr=args.redis_host
        self.conf.db_port=str(args.redis_port)
        self.mode=args.m
        self.output=args.o

    def _initialize(self) -> None:
        pass
    
    def run(self) -> None:
        if (self.mode=='io' or self.mode=='full'):
            extractor = IORequirementExtractor()
            io_requirement = extractor.run()
            if (self.mode=='io'):
                if (self.output==None):
                    JSONClient().dumps(io_requirement)
                else:
                    JSONClient().save(io_requirement, self.output)
        else:
            io_requirement = JSONClient().load(self.conf.io_req_out_path)
        if (self.mode=='stor' or self.mode=='full'):
            builder = StorageRequirementBuilder()
            storage_requirement = builder.run(io_requirement)
            if (self.mode=='stor'):
                if (self.output==None):
                    JSONClient().dumps(storage_requirement)
                else:
                    JSONClient().save(storage_requirement, self.output)
        else:
            storage_requirement = JSONClient().load(self.conf.storage_req_out_path)
        if (self.mode=='conf' or self.mode=='full'):
            configurator = StorageConfiguratorFactory.get(self.conf.storage_configurator_type)
            configuration = configurator.run(storage_requirement)
            if (self.output==None):
                JSONClient().dumps(configuration)
            else:
                JSONClient().save(configuration, self.output)
        self._finalize()
        return None

    def _finalize(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", default='full', help="Mode (options include 'io', 'stor', 'conf', and 'full')")
    parser.add_argument("-j", default=None, help="Job spec (json)")
    parser.add_argument("-t", default=None, help="Trace file (Darshan)")
    parser.add_argument("-i", default="sample/io_req_output.json", help="I/O Requirements file/template")
    parser.add_argument("-s", default="sample/stor_req_output.json", help="Storage Requirements file/template")
    parser.add_argument("-c", default="sample/stor_req_conf_output.json", help="Storage Configuration template")
    parser.add_argument("-o", default=None, help="Output file")
#     parser.add_argument(
#         "-r",
#         "--enable-redis",
#         help="""Enable use of
# Redis""",
#         action="store_true",
#     )
    parser.add_argument(
        "--redis-host", default="localhost", help="Host to use for redis"
    )
    parser.add_argument("--redis-port", default=6379, help="Port to use for redis")
    # parser.add_argument(
    #     "--redis-db", default=0, help="Database number to use for redis"
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     help="Toggle extra detail in\noutput",
    #     action="store_true",
    # )
    args = parser.parse_args()

    tool = LuxioBin(args)

    tool.run()
    exit(0)
