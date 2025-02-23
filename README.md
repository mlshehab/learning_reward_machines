# Introduction 


Code for the paper: [Learning Reward Machines From Partially Observed Optimal Policies](https://arxiv.org/pdf/2502.03762). 



<video width="640" height="360" controls>
  <source src="./video/lrm_panda_arm_stacking.mov" type="video/mov">
</video>

> **Note:** The simulation is done using the [CoppeliaSim](https://www.coppeliarobotics.com/) Simulator and [Franka Emika Panda robot](https://frankaemika.github.io/docs/index.html). The control is done using [PyRep](https://github.com/stepjam/PyRep).

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

