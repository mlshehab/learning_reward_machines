from reward_machine.reward_machine import RewardMachine
from collections import deque
import pprint
import xml.etree.ElementTree as ET

def get_label(state):
    """
    Generate a labeling string for a given state of blocks.

    Args:
        state (tuple): A tuple of tuples representing the piles of blocks.

    Returns:
        str: A string representing the labels 'G', 'Y', 'R' combined if conditions are met.
    """
    labels = []

    # Check if the green block is at the bottom of the last pile
    if state[3] and state[3][0] == 0:
        labels.append('G')

    # Check if a yellow block is on top of a green block in any pile
    if any(
        0 in pile and pile.index(0) + 1 < len(pile) and pile[pile.index(0) + 1] == 1
        for pile in state
        if len(pile) > 1 and 0 in pile
    ):
        labels.append('Y')

    # Check if a red block is on top of a yellow block in any pile
    if any(
        1 in pile and pile.index(1) + 1 < len(pile) and pile[pile.index(1) + 1] == 2
        for pile in state
        if len(pile) > 1 and 1 in pile
    ):
        labels.append('R')

    return '&'.join(labels) if labels else 'I'



def u_from_obs(obs_str, rm):
    # obs_traj : 'l0l1l2l3 ...'

    # Given the observation trajectory, find the current reward machine state

    u0 = rm.u0
    current_u = u0
    
    parsed_labels = obs_str.split(',')

    # print(f"The parsed labels are: {parsed_labels}")

    for l in parsed_labels:
        current_u = rm._compute_next_state(current_u,l)

    return current_u


def print_tree_to_text(node, file, level=0):
        # Write the node information to the file in plain text
        file.write(" " * (level * 4) + f"Node({node.label}, {node.u}, {node.policy})\n")
        for child in node.children:
            print_tree_to_text(child, file, level + 1)

   
def save_tree_to_text_file(root_node, filename):
    with open(filename, 'w') as file:
        print_tree_to_text(root_node, file)


def remove_consecutive_duplicates(s):
    """
    Removes consecutive duplicates from a comma-separated string while preserving order.
    
    Parameters:
        s (str): The input string with elements separated by commas.
    
    Returns:
        str: A string with consecutive duplicates removed, separated by commas.
    """
    elements = s.split(',')
    if not elements:
        return s  # Handle the edge case of an empty string

    result = [elements[0]]  # Start with the first element
    for i in range(1, len(elements)):
        if elements[i] != elements[i - 1]:  # Check if current is not equal to previous
            result.append(elements[i])
    
    return ','.join(result)


def collect_state_traces_iteratively(root):
    """
    Iteratively traverse the tree using BFS, starting from the root node,
    and collect proposition traces leading to each MDP state.

    Parameters:
    - root: The root node of the tree.

    Returns:
    - state_traces: A dictionary where keys are states and values are lists of proposition traces.
    """
    # Initialize the dictionary to store traces for each state
    state_traces = {}
    total_number_of_traces = 0
    # Initialize the queue with the root node
    queue = deque([root])

    # Perform BFS traversal
    while queue:
        # Dequeue the next node
        current_node = queue.popleft()

        # If the node has a valid state, use the node's label to represent the trace
        if current_node.state is not None:
            if current_node.state not in state_traces:
                state_traces[current_node.state] = []  # Initialize the list if the state is not in the dictionary
            state_traces[current_node.state].append((current_node.label,current_node.policy))
            total_number_of_traces += 1

        # Enqueue all children
        for child in current_node.children:
            queue.append(child)
    print(f"The total number of traces is: {total_number_of_traces}")
    return state_traces



def get_unique_traces(proposition_traces):
        """
        Extract unique items from a list of tuples based on the label part of each tuple.

        Parameters:
        - proposition_traces: List of tuples where each tuple contains (label, policy).

        Returns:
        - unique_traces: List of unique tuples based on the label part.
        """
        # Use a set to track unique labels  
        unique_labels = set()
        # List to store unique tuples
        unique_traces = []
        n_unique=  0
        for label, policy in proposition_traces:
            # Check if the label is already in the set
            if remove_consecutive_duplicates(label) not in unique_labels:
                # If not, add it to the set and add the tuple to the unique list
                unique_labels.add(remove_consecutive_duplicates(label))
                unique_traces.append((remove_consecutive_duplicates(label), policy))
                n_unique +=1
        return unique_traces , n_unique

def group_traces_by_policy(proposition_traces):
        """
        Group labels from proposition traces based on their policies.

        Parameters:
        - proposition_traces: List of tuples where each tuple contains (label, policy).

        Returns:
        - grouped_traces: A dictionary where keys are policies and values are lists of labels.
        """
        # Initialize a dictionary to store lists of labels for each policy
        grouped_traces = {}

        for label, policy in proposition_traces:
            # If the policy is not yet in the dictionary, add it with an empty list
            if policy not in grouped_traces:
                grouped_traces[policy] = []
            # Append the label to the corresponding policy list
            grouped_traces[policy].append(label)

        # Convert the dictionary values to separate lists
        return list(grouped_traces.values())


def write_traces_to_xml(state_traces, filename="state_traces_blockworld.xml"):
    """
    Writes the grouped traces for each state to an XML file.

    Parameters:
    - state_traces: A dictionary where each key is a state and value is a list of (label, policy) tuples.
    - n_states: The number of states.
    - filename: The output XML file name.
    """
    # Create the root element
    root = ET.Element("StateTraces")

    # for state in range(n_states):
    for state in state_traces.keys():
        # Create a state element
        state_element = ET.SubElement(root, f"State_{state}")

        # Get unique traces for the current state
        unique_traces = get_unique_traces(state_traces[state])
        # Group the traces by their policy
        grouped_lists = group_traces_by_policy(unique_traces)

        # Add grouped lists to the state element
        for idx, group in enumerate(grouped_lists, 1):
            list_element = ET.SubElement(state_element, f"List_{idx}")
            list_element.text = ", ".join(group)  # Join the items into a single string

    # Convert the ElementTree to a string
    tree = ET.ElementTree(root)
    
    # Write the XML string to a file
    tree.write(filename, encoding='utf-8', xml_declaration=True)

    print(f"Traces written to {filename}.")

# Example usage
if __name__ == '__main__':
    
    rm = RewardMachine("./rm_examples/adv_stacking.txt")
    # print(f"rm.delta_u = {rm.delta_u}")
    obs_str = 'D'
    print(u_from_obs(obs_str, rm))

    # s = 'G,G,R&Y,R&Y,R&Y&R,R&Y&R,R&Y&R,Y,I,I'
    # result = remove_consecutive_duplicates(s)
    # print(result)  # Output: 'G,R&Y,Y,I'
