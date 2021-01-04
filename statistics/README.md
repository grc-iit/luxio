# Luxio Statistics for I/O Requirement Extractor

## Objective

The objective of this project is to identify features that explain performance variables and to
classify applications that exhibit these features into categories of performance.

## Requirements

* matplotlib  
* numpy  
* pandas  
* sklearn  
* progressbar2  
* pydot  

## Structure

* preprocess.py: Creates derivative columns and removes invalid entries from the combined Mira/Theta datasets  
* analyze.py: Analyzes the preprocessed dataset in various ways  
* datasets: A directory that contains the datasets used for the analysis  
* datasets/dataset.csv: The combined trace information from Mira and Theta  
* datasets/preprocessed_dataset.csv: The dataset that contains derivative columns and no invalid entries  
* datasets/preprocessed_schema.csv: The set of columns that the preprocessed_dataset.csv contains  
* datasets/features.csv: The set of candidate features to use to describe performance  
* datasets/performance.csv: The set of performance variables we are trying to model  
* datasets/traces: Contains the unmodified trace data from Mira and Theta  

## Usage

preprocess.py has no inputs. It reads dataset/dataset.csv, removes invalid entries, and adds derivative columns.  

analyze.py has no inputs. You can look at the different cases provided at the bottom of the file.  
