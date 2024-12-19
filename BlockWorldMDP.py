import numpy as np

from scipy.special import softmax, logsumexp
import time 
from mdp import MDP, MDPRM
from itertools import permutations, product

def infinite_horizon_soft_bellman_iteration(MDP, reward,  tol = 1e-4, logging = True, log_iter = 5, policy_test_iter = 20):

    gamma = MDP.gamma
    n_actions = MDP.n_actions
    n_states = MDP.n_states
    # print(f"ns = ")

    v_soft = np.zeros((n_states,1)) # value functions
    q_soft = np.zeros((n_states, n_actions))

    delta = np.inf 

    converged = delta < tol

    it = 0
    total_time = 0.0

    while not converged:
    
        it+=1

        start_time = time.time()

        for state in range(n_states): 
            for action in range(n_actions):

                p_ns = MDP.P[action][state]

                future_value_soft = 0.0
      

                for i in range(len(p_ns)):
                    future_value_soft += p_ns[i]*reward[state][action][i] + gamma*p_ns[i]*v_soft[i]

                q_soft[state,action] =   future_value_soft
                

        v_new_soft = logsumexp(q_soft,axis = 1)

        end_time = time.time()
        total_time += end_time - start_time

        if logging and not it%log_iter and it >1 :
            print(f"Total: {it} iterations -- iter time: {end_time - start_time:.2f} sec -- Total time: {total_time:.2f}")
            print(f"Soft Error Norm ||e||: {np.linalg.norm(v_new_soft -v_soft):.6f}")
        
        
        converged = np.linalg.norm(v_new_soft - v_soft) <= tol

        v_soft = v_new_soft
    

    # find the policy
    soft_policy  = softmax(q_soft,axis = 1)
    print(f"sp = {soft_policy.shape}")

    return q_soft,v_soft , soft_policy



class BlocksWorldMDP:
    def __init__(self):
        self.colors = ["green", "yellow", "red"]
        self.num_piles = 4
        self.stacking_pile = 0  # The target pile for stacking
        self.num_actions = self.num_piles **2
        self.num_states = 0
        self.reward_target = 100
        self.reward_default = -1
        self.failure_prob = 0.0
        self.reset()

    def reset(self):
        """Initialize a random Blocks World state."""
        self.state = {
            "blocks": [
                {
                    "color": color,
                    "pile": np.random.randint(self.num_piles),
                    "height": -1,  # Placeholder, to be computed
                }
                for color in self.colors
            ]
        }
        self._update_heights()
        return self.state

    def _update_heights(self):
        """Update the height of blocks based on their pile."""
        pile_contents = {pile: [] for pile in range(self.num_piles)}
        for block in self.state["blocks"]:
            pile_contents[block["pile"]].append(block)
        
        for pile, blocks in pile_contents.items():
            # Sort blocks by height (lowest to highest)
            blocks.sort(key=lambda block: block["height"])
            for height, block in enumerate(blocks):
                block["height"] = height

    def _render_state(self):
        """Render the state as a string."""
        piles = {i: [] for i in range(self.num_piles)}
        for block in self.state["blocks"]:
            piles[block["pile"]].append((block["color"], block["height"]))  # Use color and height
        
        state_str = ""
        for pile in range(self.num_piles):
            if piles[pile]:
                pile_str = "-".join(f"{color[0]}({height})" for color, height in sorted(piles[pile], key=lambda x: x[1]))
            else:
                pile_str = "-"
            state_str += f"Pile {pile}: {pile_str}\n"
        print(state_str)

    def step(self, action):
        """
        Perform the given action and return (next_state, reward, done).
        
        Action is encoded as an integer:
            action = from_pile * num_piles + to_pile
        """
        from_pile = action // self.num_piles
        to_pile = action % self.num_piles

        print(f"Action: Move block from Pile {from_pile} to Pile {to_pile}")
        
        if from_pile == to_pile or np.random.rand() < self.failure_prob:
            # Invalid action or failed action
            print("Action failed or invalid. No change in state.")
            return self.state, self.reward_default, False

        # Find the top block in the from_pile
        moving_block = None
        highest_height = -1
        for block in self.state["blocks"]:
            if block["pile"] == from_pile and block["height"] > highest_height:
                moving_block = block
                highest_height = block["height"]

        if moving_block is None:
            # No block to move
            print("No block to move from the specified pile.")
            return self.state, self.reward_default, False

        # Move the block
        moving_block["pile"] = to_pile

        # Set the height of the moving block to be the top of the target pile
        new_pile_blocks = [block for block in self.state["blocks"] if block["pile"] == to_pile]
        moving_block["height"] = len(new_pile_blocks)  # Length determines the next height

        # Update heights to maintain consistency in all piles
        self._update_heights()

        # Compute reward
        stacking_pile_blocks = sorted(
            [block for block in self.state["blocks"] if block["pile"] == self.stacking_pile],
            key=lambda b: b["height"]
        )

        correct_order = ["green", "yellow", "red"]
        if len(stacking_pile_blocks) == len(self.colors) and \
           all(block["color"] == target_color for block, target_color in zip(stacking_pile_blocks, correct_order)):
            reward = self.reward_target
        else:
            reward = self.reward_default

        return self.state, reward, False

    def get_actions(self):
        """Return the list of all possible actions."""
        actions = []
        for from_pile in range(self.num_piles):
            for to_pile in range(self.num_piles):
                if from_pile != to_pile:
                    actions.append(from_pile * self.num_piles + to_pile)
        return actions
    
    def extract_transition_matrices(self):
        """
        Generate transition matrices for the MDP.
        Each action has a separate transition matrix.
        Rows represent current states, and columns represent next states.
        """
        num_states = self.num_piles ** len(self.colors)
        self.num_states = num_states
        transition_matrices = np.zeros((self.num_actions, num_states, num_states))

        # Map states to indices
        state_to_index = {}
        index_to_state = {}

        state_counter = 0
        for piles in range(self.num_piles ** len(self.colors)):
            state = []
            temp = piles
            for _ in range(len(self.colors)):
                state.append(temp % self.num_piles)
                temp //= self.num_piles

            state_to_index[tuple(state)] = state_counter
            index_to_state[state_counter] = tuple(state)
            state_counter += 1

        # Populate transition matrices
        for action in range(self.num_actions):
            from_pile = action // self.num_piles
            if from_pile == 3:
                print("Take from 3")
            to_pile = action % self.num_piles

            for state_index, state in index_to_state.items():
                if from_pile not in state:
                    # No block to move from this pile
                    transition_matrices[action, state_index, state_index] += 1
                    continue
                
                # Perform the action
                new_state = list(state)
                moving_block_index = state.index(from_pile)
                new_state[moving_block_index] = to_pile
                new_state_tuple = tuple(new_state)

                if new_state_tuple in state_to_index:
                    next_state_index = state_to_index[new_state_tuple]
                    transition_matrices[action, state_index, next_state_index] += (1 - self.failure_prob)
                    transition_matrices[action, state_index, state_index] += self.failure_prob

        return transition_matrices, state_to_index, index_to_state


    def extract_transition_matrices_v2(self):
        """
        Generate transition matrices for the MDP where block order in each pile matters.
        Each action has a separate transition matrix.
        Rows represent current states, and columns represent next states.
        """
        # Generate all possible placements of blocks into piles
        num_blocks = len(self.colors)
        all_states = []

        # Step 1: Generate all possible placements of blocks into piles
        for placement in product(range(self.num_piles), repeat=num_blocks):

            for permuted_piles in permutations(range(num_blocks)):
                # this iterates over all permutations, e..g (0,1,2) - (0,2,1) - etc. 
                piles = [[] for _ in range(self.num_piles)]

                for block_idx, pile_idx in zip(permuted_piles, placement):
                    piles[pile_idx].append(block_idx)
                  
                state = tuple(tuple(pile) for pile in piles)

                all_states.append(state)

        # all_states = list(set(all_states))

        self.num_states, num_states = len(all_states), len(all_states)
        state_to_index = {}
        idxx = 0
        for state in all_states:
            if state not in state_to_index.keys():
                state_to_index[state] = idxx
                idxx+=1
       
        index_to_state = {idx: state for state, idx in state_to_index.items()}

        self.num_states, num_states = len(state_to_index.keys()), len(state_to_index.keys())

        # Initialize transition matrices
        transition_matrices = np.zeros((self.num_actions, num_states, num_states))

        # Populate transition matrices
        for action in range(self.num_actions):
            from_pile = action // self.num_piles
            to_pile = action % self.num_piles

            for state_index, state in index_to_state.items():
                current_piles = [list(pile) for pile in state]  # Convert to mutable lists

                # Find the block to move
                if not current_piles[from_pile]:
                    # No block to move; self-transition
                    transition_matrices[action, state_index, state_index] += 1
                    continue

                # Move the top block
                moving_block = current_piles[from_pile].pop()
                current_piles[to_pile].append(moving_block)

                # Convert back to a tuple state representation
                new_state = tuple(tuple(pile) for pile in current_piles)

                if new_state in state_to_index:
                    next_state_index = state_to_index[new_state]
                    transition_matrices[action, state_index, next_state_index] += (1 - self.failure_prob)
                    transition_matrices[action, state_index, state_index] += self.failure_prob
                else:
                    # Self-transition if the state isn't valid (unlikely here)
                    transition_matrices[action, state_index, state_index] += 1

        return transition_matrices, state_to_index, index_to_state


    def _cartesian_product(self, list_of_lists):
        """Generate Cartesian product from a list of lists."""
        result = [[]]
        for lst in list_of_lists:
            result = [x + [y] for x in result for y in lst]
        return result


# Example usage
if __name__ == "__main__":

    env = BlocksWorldMDP()
    state = env.reset()
    # print("Initial State:")
    # env._render_state()

    # for _ in range(10):
    #     action = np.random.choice(env.get_actions())
    #     next_state, reward, done = env.step(action)
    #     print(f"Action: {action}, Reward: {reward}")
    #     print("Next State:")
    #     env._render_state()

    # Extract transition matrices
    transition_matrices,s2i, i2s = env.extract_transition_matrices_v2()
    print("Transition Matrices Shape:", transition_matrices.shape)
    print(f"we have: {len(s2i.keys())} keys in i2s.")

    # for k in i2s.items():
    #     print(k)
    n_states = env.num_states
    # print(f"The n sss is: {n_states}")
    n_actions = env.num_actions

    P = []

    for a in range(n_actions):
        # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

    reward = np.zeros((n_states, n_actions, n_states))  

    dst = ((),(),(),(0,1,2))
    desired_state_index = s2i[dst]
    # print(f"des = {desired_state_index}")
    # print(f"The desired state is: {desired_state_index}")

    for a in range(n_actions):
        for s_prime in range(n_states):
            reward[desired_state_index,a,s_prime] = 10.0

    # q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp,reward,logging = True)

    # # print(f"The policy is: {soft_policy.shape}")

    # np.save("soft_policy.npy", soft_policy)

    soft_policy = np.load("soft_policy.npy")

    state = ((0, 2), (), (), (1,))

    for i in range(5):
        ss2i = s2i[state]
        action = np.argmax(soft_policy[ss2i])
        # action = 0
        print(f"The action is: {action}")

        next_state_index = np.where(transition_matrices[action,ss2i,:]>0)[0][0]
        next_state_tuple = i2s[next_state_index]
      
        print(f"Next state is: {next_state_tuple}")
        state = next_state_tuple

    
    #------------------------------------------------------------------

    # n_states = env.num_states
    # # print(f"The n sss is: {n_states}")
    # n_actions = env.num_actions

    # P = []

    # for a in range(n_actions):
    #     # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
    #     P.append(transition_matrices[a,:,:])

    # mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

    # reward = np.zeros((n_states, n_actions, n_states))  

    # desired_state_index = s2i[(2,2,1)]
    # # print(f"The desired state is: {desired_state_index}")

    # for a in range(n_actions):
    #     for s_prime in range(n_states):
    #         reward[desired_state_index,a,s_prime] = 10.0

    # q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp,reward,logging = False)

    # print(f"The policy is: {soft_policy.shape}")

    

    # state = (1,1,0)

    # for i in range(15):
    #     ss2i = s2i[state]
    #     action = np.argmax(soft_policy[ss2i])
    #     print(f"The action is: {action}")

    #     from_pile = action // env.num_piles
    #     to_pile = action % env.num_piles

    #     if from_pile not in state:
    #         print("Done!")
    #         continue 

    #     # Perform the action
    #     new_state = list(state)
    #     moving_block_index = state.index(from_pile)
    #     new_state[moving_block_index] = to_pile
    #     new_state_tuple = tuple(new_state)
    #     print(f"Next state is: {new_state_tuple}")
    #     state = new_state_tuple