import json
import argparse
import numpy as np
from mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from scipy.special import softmax, logsumexp
from tqdm import tqdm
import pprint
import xml.etree.ElementTree as ET
from collections import deque
from BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from ne_utils import get_label, u_from_obs,save_tree_to_text_file, collect_state_traces_iteratively, get_unique_traces,group_traces_by_policy
from sat_utils import *
from datetime import timedelta


# Define the configurations for different experiments
CONFIGURATIONS = {
    "exp1": {
        "grid_size": [5, 5],
        "labeling_function": "label_func_1.json",
        "reward_machine_path": "rm_1.json",
        "other_param": 42
    },
    "exp2": {
        "grid_size": [10, 10],
        "labeling_function": "label_func_2.json",
        "reward_machine_path": "rm_2.json",
        "other_param": 84
    },
    "exp3": {
        "grid_size": [7, 7],
        "labeling_function": "label_func_3.json",
        "reward_machine_path": "rm_3.json",
        "other_param": 21
    },
    "exp4": {
        "grid_size": [15, 15],
        "labeling_function": "label_func_4.json",
        "reward_machine_path": "rm_4.json",
        "other_param": 100
    }
}

def load_config(experiment_name):
    """Loads the configuration for a given experiment."""
    if experiment_name not in CONFIGURATIONS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return CONFIGURATIONS[experiment_name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment name (exp1, exp2, exp3, exp4)")
    args = parser.parse_args()
    
    config = load_config(args.experiment)
    # print(f"Running experiment {args.experiment} with config: {config}")
    

    if args.experiment == 'exp4':
        bw = BlocksWorldMDP(num_piles=3)

        transition_matrices,s2i, i2s = bw.extract_transition_matrices_v2()
        n_states = bw.num_states
        n_actions = bw.num_actions

        # print(bw)

        P = []

        for a in range(n_actions):
            # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
            P.append(transition_matrices[a,:,:])

        mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

        rm = RewardMachine("./rm_examples/adv_stacking.txt")
        # print(f"rm.delta_u = {rm.delta_u}")


        policy = {}
        for rms in range(rm.n_states):
            policy[rms] = f"p{rms}"
        
        policy[2] = policy[3]

        # print("The policy is: ", policy)
    
        L = {}

        # print(f"The number of states is: {len(s2i.keys())}")

        # for state_index in range(bw.num_states):
        #     state_tuple = i2s[state_index]
        #     L[state_index] = get_label(state_tuple)

        target_state_1 = ((0,1,2),(),())
        target_state_2 = ((),(2,1,0),())
        target_state_3 = ((),(),(2,1,0))
        bad_state = ((0,),(1,),(2,))


        for state_index in range(bw.num_states):
            if state_index == s2i[target_state_1]:
                L[state_index] = 'A'
            elif state_index == s2i[target_state_2]:
                L[state_index] = 'B'
            # elif state_index == s2i[target_state_3]:
            #     L[state_index] = 'C'
            elif state_index == s2i[bad_state]:
                L[state_index] = 'D'
            else:
                L[state_index] = 'I'
            

   
    
if __name__ == "__main__":
    main()
