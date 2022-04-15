# LUXIO

A tool for submitting I/O requirements to Flux scheduler.

## Installation

To install Luxio

> cd /path/to/luxio  
> pip3 install -r requirements.txt  
> python3 setup.py sdist bdist_wheel  
> pip3 install dist/*.whl  
> rm -r dist build *.egg_info MANIFEST

To remove Luxio:

> pip3 uninstall luxio

## Dependencies

Luxio depends on the following:
* Darshan
* Redis

### Darshan

> git clone https://xgitlab.cels.anl.gov/darshan/darshan   
> cd /path/to/darshan   
> cd darshan-util  
>  ./configure --prefix=</path/to/wherever-you-want> --enable-pydarshan --enable-shared  
> make install

## Usage

Import the module:
> from luxio import *

Configure the tool:
> conf = ConfigurationManager.get_instance()  
> conf.job_spec = "/path/to/job_spec.json"  
> conf.darshan_trace_path = "/path/to/darshan_trace.darshan"  
> conf.io_req_out_path = "/path/to/io_req_out.json"  
> conf.storage_req_out_path = "/path/to/storage_req_out.json"  
> conf.storage_req_config_out_path = "/path/to/storage_req_config_out.json"
> conf.db_type = KVStoreType.REDIS  
> conf.db_addr = "127.0.0.1"  
> conf.db_port = "6379"  

Get the I/O configuration
> storage_config = LuxioBin().run()

## Test

cd /path/to/luxio  
export PYTHONPATH="$(pwd)/src"  
> python3 -m unittest discover -s test/integration/luxio  

> ./luxio.py -m io -o output_file.json -j job_info.json -t resources.darshan  
> ./scripts/print-luxio.sh output_file.json  
The print-luxio.sh script requires the program jq (json query) to be installed, it performs a `jq 'map_values(.val)'` on the given argument

Alternatively, there are now convenience scripts to accomplish the same results as above. Try running:
> cd scripts
> source init-luxio.sh

The scripts in the scripts/ directory include luxio.sh, io-req-extractor.sh, storage-requirement-builder.sh, and storage-configurator.sh, and implement simple wrappers over modes of the Luxio system from luxio.py.

### Script Run Sample Interaction
> cd scripts
> source init-luxio.sh
> ./luxio.sh ../resources/job_info.json ../sample/sample.darshan
> ./luxio.sh ../resources/job_info.json ../sample/sample.darshan -o output-0.json
> ./io-req-extractor.sh ../resources/job_info.json ../sample/sample.darshan
> ./io-req-extractor.sh ../resources/job_info.json ../sample/sample.darshan -o output-1.json
> ./storage-requirement-builder.sh output-1.json
> ./storage-requirement-builder.sh output-1.json output-2.json
> ./storage-configurator.sh ../resources/job_info.json output-1.json output-2.json
> ./storage-configurator.sh ../resources/job_info.json output-1.json output-2.json output-3.json
> diff output-0.json output-3.json

## License
