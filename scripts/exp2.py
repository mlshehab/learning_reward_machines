import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from dynamics.GridWorld import BasicGridWorld
import numpy as np
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from scipy.special import softmax, logsumexp
from tqdm import tqdm
import pprint
import xml.etree.ElementTree as ET
from collections import deque
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from utils.ne_utils import get_label, u_from_obs,save_tree_to_text_file, collect_state_traces_iteratively, get_unique_traces,group_traces_by_policy
from utils.sat_utils import *
from datetime import timedelta

class Node:
    def __init__(self, label,state, u, policy, is_root = False):
        self.label = label
        self.parent = None
        self.state = state
        self.u = u
        self.policy = policy
        self.children = []
        self.is_root = is_root

    def __str__(self):
        return f"Node with ({self.label},{self.state}) and Parent's label is {self.parent.label}"
    
    # Method to add a child to the node
    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child node
        # self.children = []
        self.children.append(child_node)  # Add the child node to the children list

    # Method to get the parent of the node
    def get_parent(self):
        return self.parent

    # Method to get the children of the node
    def get_children(self):
        return self.children
    

def get_future_states(s, mdp):
    P = mdp.P
    post_state = []
    for Pa in P:
        for index in np.argwhere(Pa[s]> 0.0):
            post_state.append(index[0])     
    return list(set(post_state))

def get_future_states_action(s,a, mdp):
    Pa = mdp.P[a]
    post_state = []
    
    for index in np.argwhere(Pa[s]> 0.0):
        post_state.append(index[0])   

    return list(set(post_state))



import argparse

# Define a function to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Automate script with depth option")
    parser.add_argument("-depth", type=int, help="Set the depth", required=True)
    return parser.parse_args()

if __name__ == '__main__':

    grid_size = 3
    wind = 0.1
    discount = 0.9
    horizon =   13   
    start_state = 4 
    feature_dim = 2

    theta = 5*np.ones((2,1))
    theta[1] += 5
    p1,p2 = 0,24
    feature_type = 'None'

    gw = BasicGridWorld(grid_size,wind,discount,horizon,start_state, \
                         feature_dim, p1,p2,theta, feature_type)
    
    n_states = gw.n_states
    n_actions = gw.n_actions

    P = []

    for a in range(n_actions):
        P.append(gw.transition_probability[:,a,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = gw.discount,horizon=10)
    
   
    rm = RewardMachine("../rm_examples/patrol.txt")
    print(f"rm.delta_u = {rm.delta_u}")
    policy = {}
    for rms in range(rm.n_states):
        policy[rms] = f"p{rms}"
    
   

    print("The policy is: ", policy)
    # print(rm.delta_u)

 
    # The grid numbering and labeling is :
    # 0 3 6     D H C 
    # 1 4 7     H H H 
    # 2 5 8     A H B
     
    L = {}
   
    L[2]     = 'A'
    L[8] = 'B'
    L[6]= 'C'
    L[0]     = 'D'
    L[3], L[4], L[5],L[1],L[7] = 'H','H','H','H','H'

    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
    # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    # print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")


                
    # #############
    # #############
    # # PREFIX TREE
    # #############
    # #############

    # Parse command-line arguments
    args = parse_args()
    
    # Set the depth variable from the command line argument
    depth = args.depth
    print(f"The depth is: {depth}")
    Root = Node(label = None, state= None, u = None,policy = None , is_root= True)
    queue = [(Root, 0)]  # Queue of tuples (node, current_depth)

    # The first time step here is assuming a fully supported starting distribution
    current_node, current_depth = queue.pop(0)  # Dequeue the next node
    

    # test_state = ((),(1,),(2,),(0,))
    # print(f"The state index of interest is: {s2i[test_state]}")
    # for s in get_future_states(s2i[test_state], mdp):
    #     print(f"The future states are: {i2s[s]}, L = {L[s]}")
    starting_states = [0,2,3,6,8]
    for s in starting_states:
    # for s in range(mdp.n_states):
        # get label of the state
        label = L[s]
        # create a node for that state
        child_node = Node(label = label+',',state = s, u = u_from_obs(label,rm), policy = None)
        child_node.policy = policy[child_node.u]
        current_node.add_child(child_node)
        queue.append((child_node, current_depth + 1))

    while queue:
        current_node, current_depth = queue.pop(0)  # Dequeue the next node

        if current_depth < depth:
            # get the state of the current node
            s = current_node.state
            # get the future possible states
            next_states = get_future_states(s,mdp)

            for nx_s in next_states:
                # get label of the next state
                
                label = L[nx_s]
             
                # create a node for that state
                nx_u = u_from_obs(current_node.label + label, rm)
                if nx_u == -1:
                    raise ValueError("Got a next state of -1.")
                if current_depth == depth - 1:
                    mod_label = label
                else:
                    mod_label = label + ','
                child_node = Node(label = current_node.label + mod_label, state = nx_s, u = nx_u, policy = None)
                child_node.policy = policy[child_node.u]
                current_node.add_child(child_node)
                queue.append((child_node, current_depth + 1))
    
  
    # save_tree_to_text_file(Root, 'BlockWorldTree.txt')

    # # Example usage
    state_traces = collect_state_traces_iteratively(Root)


    state_traces_dict = {}

    total_number_of_unique_traces = 0
    for state in state_traces.keys():
        # Get unique traces for the current state
        unique_traces, n_unique = get_unique_traces(state_traces[state])
        total_number_of_unique_traces += n_unique
        # Group the traces by their policy
        grouped_lists = group_traces_by_policy(unique_traces)

        state_traces_dict[state] = grouped_lists

    print(f"The total number of Unique traces is: {total_number_of_unique_traces}")


    ###############################################
    ###### SAT Problem Encoding Starts HERE #######
    ###############################################

    kappa = 4
    AP = 5
    total_variables = kappa**2*AP
    total_constraints = 0

    B = [[[Bool('x_%s_%s_%s'%(i,j,k) )for j in range(kappa)]for i in range(kappa)]for k in range(AP)]

    B_ = element_wise_or_boolean_matrices([b_k for b_k in B])
    x = [False]*kappa
    x[0] = True
    print(f"x = {x}")

    B_T = transpose_boolean_matrix(B_)

    powers_B_T = [boolean_matrix_power(B_T,k) for k in  range(1,kappa)]
    
    powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    
    powers_B_T_x.insert(0, x)
    
    # print(powers_B_T_x[0])
    OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    # print(OR_powers_B_T_x)
    s = Solver() # type: ignore

    # C0 Trace compression
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                # For boolean variables, B[ap][i][j], add the constraint that the current solution
                # is not equal to the previous solution
                s.add(Implies(B[ap][i][j], B[ap][j][j]))
                # total_constraints +=1


    # C1 and C2 from Notion Write-up
    for k in range(AP):
        # total_constraints +=1
        s.add(one_entry_per_row(B[k]))


    
    proposition2index = {'A':0, 'B':1 , 'C':2, 'D':3, 'H':4}

    def prefix2indices(s):
        # print(f"The input string is: {s.split(',')}")
        out = []
        for l in s.split(','):
            if l:
                out.append(proposition2index[l])
        return out


    counter_examples = generate_combinations(state_traces_dict)

    # print(f"The type is :{type(counter_examples)}")

    # C4 from from Notion Write-up 
    print("Started with C4 ... \n")
    total_start_time = time.time()


    print(f"We have a total of {len(counter_examples.keys())} states that give negative examples.")
    all_ce = []
    for state in counter_examples.keys():
        print(f"Currently in state {state}...")
        ce_set = counter_examples[state]
        # print(f"The ce_set is: {ce_set}")
        print(f"The number of counter examples is: {len(ce_set)}\n")
        total_constraints += len(ce_set)
        
        # for each counter example in this set, add the correspodning constraint
        for ce in tqdm(ce_set,desc="Processing Counterexamples"):
            all_ce.append(ce)
            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])

            # Now
            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)

            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            for elt in res_:
                s.add(Not(elt))
                
        
    print(f"we have a tortal of {total_constraints} constraints!")

    # Open the file in write mode
    with open('output_2S.txt', 'w') as file:
        for item1, item2 in all_ce:
            # Remove commas from both items
            item1 = item1.replace(',', '')
            item2 = item2.replace(',', '')
            
            # Write the processed pair to the file, joined by a space
            file.write(f"{item1} {item2}\n")

    # Use timedelta to format the elapsed time
    elapsed  = time.time() - total_start_time
    formatted_time = str(timedelta(seconds= elapsed))

    # Add milliseconds separately
    milliseconds = int((elapsed % 1) * 1000)

    # Format the time string
    formatted_time = formatted_time.split('.')[0] + f":{milliseconds:03}"
    print(f"Adding C4 took {formatted_time} seconds.")

    
    # Start the timer
    start = time.time()
    s_it = 0
    while True:
        # Solve the problem
        if s.check() == sat:
            end = time.time()
            print(f"The SAT solver took: {end-start} sec.")
            # Get the current solution
            m = s.model()
            
            # # Store the current solution
            # solution = []
            print(f"Solution {s_it} ...")
            for ap in range(AP):
                r = [[m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
                # solution.append(r)
                
                print_matrix(r)  # Assuming print_matrix prints your matrix nicely
            s_it += 1
            # # Add the solution to the list of found solutions
            # solutions.append(solution)

            # Build a clause that ensures the next solution is different
            # The clause is essentially that at least one variable must differ
            block_clause = []
            for ap in range(AP):
                for i in range(kappa):
                    for j in range(kappa):
                        # For boolean variables, B[ap][i][j], add the constraint that the current solution
                        # is not equal to the previous solution
                        block_clause.append(B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True))

            # Add the blocking clause to the solver
            s.add(Or(block_clause))
            
        else:
            print("NOT SAT - No more solutions!")
            break


