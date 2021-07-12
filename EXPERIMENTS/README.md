# Experiments for In-Memory ANN Search Algorithms

## Usage

+ Formating datasets
```bash
./prepare.py
```

+ Run experiments
```bash
Usage: run.sh [-ahr]

This script attempts to run ANN experiments
options:
 -a: (default) run (A)ll algorithms - good luck!
 -c: clean all!
 -h: print this (H)elp message
 -r <alg>: only run <alg>, where <alg>=LinearScan|C2LSH|iDEC|QALSH|SRS
```
