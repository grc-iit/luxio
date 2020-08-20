#!/usr/bin/env python3

import argparse
from common.error_codes import *
from common.enumerations import *
from common.configuration_manager import *
from io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from storage_requirement_builder.storage_requirement_builder import *
from storage_configurator.storage_configurator_factory import *
from external_clients.json_client import *
from utils.argument_parser import *

class LuxioBin:
    def __init__(self, conf):
        self.conf = conf

    def _initialize(self) -> None:
        pass
    
    def run(self) -> None:
        if (self.conf.run_mode=='io' or self.conf.run_mode=='full'):
            extractor = IORequirementExtractor()
            io_requirement = extractor.run()
            if (self.conf.run_mode=='io'):
                if (self.conf.output_file==None):
                    JSONClient().dumps(io_requirement)
                else:
                    JSONClient().save(io_requirement, self.conf.output_file)
        else:
            io_requirement = JSONClient().load(self.conf.io_req_out_path)
        if (self.conf.run_mode=='stor' or self.conf.run_mode=='full'):
            builder = StorageRequirementBuilder()
            storage_requirement = builder.run(io_requirement)
            if (self.conf.run_mode=='stor'):
                if (self.conf.output_file==None):
                    JSONClient().dumps(storage_requirement)
                else:
                    JSONClient().save(storage_requirement, self.conf.output_file)
        else:
            storage_requirement = JSONClient().load(self.conf.storage_req_out_path)
        if (self.conf.run_mode=='conf' or self.conf.run_mode=='full'):
            configurator = StorageConfiguratorFactory.get(self.conf.storage_configurator_type)
            configuration = configurator.run(storage_requirement)
            if (self.conf.output_file==None):
                JSONClient().dumps(configuration)
            else:
                JSONClient().save(configuration, self.conf.output_file)
        self._finalize()
        return None

    def _finalize(self) -> None:
        pass


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    conf = arg_parser.run()
    tool = LuxioBin(conf)
    tool.run()

    exit(0)
