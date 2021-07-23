# Experiments for In-Memory ANN Search Algorithms

## Please install Highfive(https://github.com/BlueBrain/HighFive) first before running the experiments.

## Usage

+ Formating datasets
```bash
./prepare.py
```

+ Run experiments
```bash
Usage: run_l1.sh [-ahr]

This script attempts to run ANN experiments
options:
 -a: (default) run (A)ll algorithms - good luck!
 -c: clean all!
 -h: print this (H)elp message
 -r <alg>: only run <alg>, where <alg>=LinearScan|FALCONN|FALCONN_cauchy|FALCONN_RW
```
## Running Example
```bash
./run_l1.sh -r LinearScan
./run_l1.sh -r FALCONN
```
Make sure to run **LinearScan** before running **FALCONN** algorithms.

### Algorithms name in paper
FALCONN: MP-RW-LSH

FALCONN_cauchy: CP-LSH

FALCONN_RW: RW-LSH
