# Introduction 

[![Paper Status](https://img.shields.io/badge/RA--L-Under%20Review-orange)](https://www.ieee-ras.org/publications/ra-l)

Code for the paper: "Learning Reward Machines From Partially Observed Optimal Policies", submitted to RA-L journal. 

# Structure

This repository contains a set of experiment scripts inside the `scripts` directory. To easily run these experiments, use the `run.sh` script.

## Running an Experiment

To execute an experiment, use the `run.sh` script from the root of the repository:

### Example Usage
```
./run.sh exp2 -depth 5
```

This will:
- Change into the `scripts` directory.
- Run the `exp2.py` script with the provided argument (`-depth 5`).


## Requirements

Ensure you have Python installed and that the `run_experiment.sh` script has execution permissions:
```
chmod +x run_experiment.sh
```

## Available Experiments

The following experiments are currently available in the `scripts` folder:

- `exp1.py`
- `exp2.py`
- `exp3.py`
- `exp4.py`

To add a new experiment, simply place the corresponding Python script inside the `scripts` directory.

## Notes

- You can pass additional arguments to the script, and they will be forwarded to the experiment script.
- If an invalid experiment name is given, the script will return an error.

