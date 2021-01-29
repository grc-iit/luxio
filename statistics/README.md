# Luxio Statistics for I/O Requirement Extractor

## Objective

The objective of this project is to identify features that explain performance variables and to
classify applications that exhibit these features into categories of performance.

## Requirements

* clever 0.0.0

## Usage

```{bash}
python3 analyze.py -t [tool] -c [conf]  
```

[tool]:  
* preprocess: preprocess data from ANL  
* fmodel: Feature reduction model  
* fmodel_stats: Statistics about the reduced model  
* bmodel: Behavior classifier using the stats  
* bmodel_stats: Statistics about the behavior classification  

An example configuration file is under /conf  
