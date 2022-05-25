# LUXIO

A tool for autconfiguring and provisioning storage.

## Dependencies

Luxio depends on the following:
* SCSPKG (a tool for managing modulefiles)
* Darshan
* Redis

## Installation

### Docker Container
We have a dockerfile to make deployment easier
```
git clone https://github.com/scs-lab/luxio.git
cd luxio
sudo docker build -t luxio-img .
sudo docker run -it --name luxio luxio-img
```

### For Regular Users
```{bash}
git clone https://github.com/scs-lab/luxio.git
cd luxio
bash dependencies.sh
python3 -m pip . --user
module load luxio
```

### For Developers

```{bash}
git clone git@github.com:scs-lab/luxio.git
cd luxio
python3 -m pip install -e .
```

### Uninstallation

```{bash}
python3 -m pip uninstall luxio
```

## Usage

Start the luxio server to maintain resource graph:
```{bash}
luxio-server --conf [/path/to/luxio-conf.json]
```

Run the Luxio scheduler assistance command line tool:
```{bash}
luxio-sched --conf [/path/to/luxio-conf.json]
```

A sample conf is in sample/luxio_confs

### Datasets

Our datasets can be downloaded from here:
```
mkdir datasets
cd datasets
#Argonne Dataset
gdown https://drive.google.com/uc?id=1rLOmmFz6TexGDgIaR6qgOp_FMoEJjAeV
#SSLO data parquet
gdown https://drive.google.com/uc?id=1Rgg3S7gVW48L28dEQiA5p8ZWKisNutPk
```

## Test

In order to run tests, you must have installed Luxio in either developer mode or regular mode.

Before running tests:
```{bash}
#Start Redis
redis-server
#Upload application and storage behavior models
luxio-stats -t upload
#Start luxio server
luxio-server --conf [/path/to/luxio-conf.json]
```

To run all tests:
```{bash}
cd /path/to/luxio
pytest
```

To run only unit tests:
```{bash}
cd /path/to/luxio
pytest -k unit -s
```

To run only integration tests:
```{bash}
cd /path/to/luxio
pytest -k integration -s
```
