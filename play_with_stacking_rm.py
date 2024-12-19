import numpy as np
from mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from scipy.special import softmax, logsumexp

import pprint
import xml.etree.ElementTree as ET
from collections import deque
from BlockWorldMDP import BlocksWorldMDP
from sat_utils import get_label

rm = RewardMachine("./rm_examples/static_stacking.txt")
# print(f"rm.delta_u = {rm.delta_u}")
for a in rm.delta_u.items():
    print(a)

labels = ['G','R','Y','G&Y','G&R','G&Y&R','I','Y&R','Y&G']
nodes = [0,1,2,3]

# for node in nodes:
#     for label in labels:
#         print(f"From u = {node}, with l = {label}, we arrive at: {rm._compute_next_state(node,label)}")

node = 0
state = ((), (), (1,2), (0,))
label = get_label(state)
print(f"From u = {node}, with l = {label}, we arrive at: {rm._compute_next_state(node,label)}")



# l = 'G'
# current_u = 0
# print(f"the next state is: {rm._compute_next_state(current_u,l)}")