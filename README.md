# LUXIO

A tool for submitting I/O requirements to Flux scheduler.

## Code Features

## Installation

## Dependencies

python3 -m pip install msgpack

## Usage

## Contribute to LUXIO

### Code examples to use:

- Factory Pattern: [here](https://github.com/hariharan-devarajan/dlio_benchmark/blob/master/src/data_generator/generator_factory.py)
- Singleton Pattern: [here](https://github.com/hariharan-devarajan/dlio_benchmark/blob/master/src/utils/argument_parser.py#L30)

### Jobs to do
Hari
- finalize all datastructures
    - job info (map)
    - darshan counters (map)
    - io_requirements (map)
    - storage_requirements (map)
    - storage configuration (map)

Keith
- utility whioch converts job spec yaml to map
- create a job speck json.
- put utility functions in src/utils/


### Code Styles

- __ for only internal python variables/methods
- _ for private variables/methods
- decorate types for readability [here](https://docs.python.org/3/library/typing.html)
- class names are same as parent package
- inheritence [here](https://github.com/hariharan-devarajan/dlio_benchmark/blob/master/src/data_generator/data_generator.py#L10)
- error codes: [here](https://github.com/hariharan-devarajan/dlio_benchmark/blob/master/src/common/error_code.py)

### Test

cd /path/to/luxio  
export PYTHONPATH="$(pwd)/src"  
> python3 test/unit/external_clients/serializer/test.py  
> python3 test/unit/external_clients/kv_store/test.py  
> python3 test/unit/io_requirement_extractor/test.py  
> python3 test/unit/mapper_manager/test.py  
> python3 test/unit/luxio/test.py  

## License
