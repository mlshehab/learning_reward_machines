
import math
import numpy as np
import scipy.special
import cvxpy as cp

inf  = float("inf")


class BasicGridWorld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount,horizon,start_state, feature_dim, p1, p2,theta, feature_type):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.wind_buffer = wind
        
        self.discount = discount
        self.horizon = horizon
        self.feature_dim = feature_dim
        self.p1 = p1
        self.p2 = p2
        
        self.blocked_states = [6,7,8,9]
        
        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        
        
        
        self.normalize_transition_matrices()
        
        # self.make_state_determinstic(1)
        # self.make_state_determinstic(2)
        # self.make_state_determinstic(3)
        # self.make_state_determinstic(4)
        
        self.theta = theta
        
        self.start_state = start_state
        self.feature_type = feature_type
#         self.make_state_determinstic(12)
        self.reward_v = self.reward_function(self.theta)

    def __str__(self):
        return "a"
    # def __str__(self):  
    #     return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
    #                                           self.discount)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1
    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0
        
    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
            
        
        # # is this state blocked in the MDP?
        # left_column = (i == 1) or (i == 2) or (i == 3) or (i == 4)
        # lc = [1,2,3,4]

        # disallow transition from the leftmost column to the one next to it
        # if k in self.blocked_states:
        #     return 0.0

        # disallow self transitions for the left column
        # if i == k and left_column:
        #     return 0.0
        
        # # disallow transition from the leftmost column to the one next to it
        # if k in self.blocked_states and left_column:
        #     return 0.0

        # # disallow transition from the 2nd column to the left most column 
        # if k in lc and i in self.blocked_states:
        #     return 0.0
        


        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions
            
    def normalize_transition_matrices(self):
        for a in range(self.n_actions):
            P = self.transition_probability[:,a,:]
            sum_P = P.sum(axis = 1)
            normalized_P = P/sum_P[:,None]
            self.transition_probability[:,a,:] = normalized_P
            
    
    def make_state_determinstic(self, s):
        
        for s_prime in range(self.n_states):
            for a in range(self.n_actions):
                self.transition_probability[s,a,s_prime] = 0.0
                
        for a in range(self.n_actions):
            self.transition_probability[s,a,s-1] = 1.0
         
            
    def generate_trajectories(self):
        # Memoization dictionary to store already computed results
        horizon = self.horizon
        memo_dict = {}

        def generate_recursive(current_time, current_state):
            # Check if the result is already computed
            if (current_time, current_state) in memo_dict:
                return memo_dict[(current_time, current_state)]

            # Base case: when we reach the horizon, return an empty trajectory
            if current_time == horizon:
                return [[]]

            trajectories = [] 

            # Recursive case: try all actions and next states
            for action in range(self.n_actions):
                next_state_probs = self.transition_probability[current_state,action,:] if current_state is not None else self.start_distribution

                for next_state in range(self.n_states):
                    # Recursive call
                    if next_state_probs[next_state] != 0:
                        next_trajectories = generate_recursive(current_time + 1, next_state)

                        # Extend the current trajectory with the current action and next state
                        if current_state == None:

                            trajectories.extend([(next_state, action )] + traj for traj in next_trajectories)
                        else:
                            trajectories.extend([(current_state, action, next_state)] + traj for traj in next_trajectories)

            # Memoize the result before returning
            memo_dict[(current_time, current_state)] = trajectories
            
            print("Length is:", len(trajectories))
            return trajectories

        # Start the recursion with time = 0 and no initial state
        # For a unique starting state
        return generate_recursive(0, self.start_state)
    
    def get_trajectory_matrix(self):
        
        trajs = self.generate_trajectories()
        
        M = np.zeros((len(trajs), self.n_states*self.n_actions ))
        
        
        for i in range(len(trajs)):
            for t,time_step in enumerate(trajs[i]):
                
                s = time_step[0]
                a = time_step[1]
                
                pos_ = a*self.n_states + s
                
                M[i][pos_] += self.discount**t
                
        return M
    
   
    def manhatten_distance(self, i,k):
        
        xi, yi = self.int_to_point(i)
        xk, yk = self.int_to_point(k)
        
        return np.abs(xi-xk) + np.abs(yi- yk)
    
    def feature_vector_v1(self,s,a):
        
        f = np.zeros((self.feature_dim,1))
        eps = 0.5
        f[0] = 1.0/(self.manhatten_distance(s, self.p1)+eps)
        
        f[1] = 1.0/(self.manhatten_distance(s, self.p2)+eps)
        
        
        
        xs, ys  = self.int_to_point(s)
        xa, ya = self.actions[a]
        
        des_ = (xs+xa,ys+ya)
        des_s = self.point_to_int(des_)
        
        if self.manhatten_distance(des_s , self.p1) <= f[0]:
            f[0] += 0.5
        else: 
            f[0] -= 0.5
        
        if self.manhatten_distance(des_s , self.p2) <= f[1]:
            f[1] += 0.5
        else: 
            f[1] -= 0.5
        
        return f

    def sparse_reward_features(self,s,a):
   
        f = np.zeros((self.feature_dim,1))

        f[0] = 1 if s == self.p1 else 0
        
        f[1] = 1 if s == self.p2 else 0
        
        return f

    def dense_reward_features(self,s,a):
        
        f = np.zeros((self.feature_dim,1))

        f[0] = -self.manhatten_distance(s, self.p1)
        
        f[1] = -self.manhatten_distance(s, self.p2)
        
        
        
        xs, ys  = self.int_to_point(s)
        xa, ya = self.actions[a]
        
        des_ = (xs+xa,ys+ya)
        des_s = self.point_to_int(des_)
        
        # if self.manhatten_distance(des_s , self.p1) <= f[0]:
        #     f[0] += 0.5
        # else: 
        #     f[0] -= 0.5
        
        # if self.manhatten_distance(des_s , self.p2) <= f[1]:
        #     f[1] += 0.5
        # else: 
        #     f[1] -= 0.5
        return f
    
    def F_matrix(self):
        feature_function = self.dense_reward_features if self.feature_type == "dense" else self.sparse_reward_features
        
        F = np.zeros((self.n_states*self.n_actions, 2))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                F[s + a*self.n_states][0] = feature_function(s,a)[0]
                F[s + a*self.n_states][1] = feature_function(s,a)[1]
        return F
    
    
    def reward_function(self, theta):
        
        feature_function = self.dense_reward_features if self.feature_type == "dense" else self.sparse_reward_features
        r_gw = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                f = feature_function(s,a)
                r_gw[s][a] = f.T@theta
                
        return r_gw


    def get_reachable_tube(self):
        # the reachable tube is a dictionary with keys 't' being the timestep. 
        Reach = {}
        UnReach = {}

        T = self.horizon
        
        Pa1 = self.transition_probability[:,0,:]
        Pa2 = self.transition_probability[:,1,:]
        Pa3 = self.transition_probability[:,2,:]
        Pa4 = self.transition_probability[:,3,:]

        P = [Pa1,Pa2,Pa3,Pa4]
        # reach at time zero
        # reach at time zero
        Reach[str(0)] = [self.start_state]
        UnReach[str(0)]= list(set(np.arange(0,self.n_states)) - set(Reach[str(0)]))

        for t in range(1,T):
            
            pre = Reach[str(t-1)]
        
            pre_vec = np.zeros((1,self.n_states))
            
            # create the vector of pre states 
            for s in pre:
                pre_vec[0][s] = 1.0
            
            # normalize
            pre_vec = pre_vec/pre_vec.sum()
            
            post_state = []
            
            for pre_state in pre:
                for Pa in P:
                    transfer = pre_vec@Pa
                    
                    for index in np.argwhere(transfer> 0.0):
                        post_state.append(index[1])
                        
            post_state = list(set(post_state))

            Reach[str(t)] = post_state
            UnReach[str(t)] = list(set(np.arange(0,self.n_states)) - set(Reach[str(t)]))


        return Reach, UnReach



    def construct_Gamma(self):
        m = self.n_actions
        n = self.n_states
        horizon = self.horizon

        I = np.eye(self.n_states)
        E = I
        P = self.transition_probability[:,0,:]

        for a in range(1, self.n_actions):
            E = np.vstack((E,I))
            P = np.vstack((P, self.transition_probability[:,a,:]))
            
        bold_I = np.eye(self.n_actions*self.n_states)
        
        Gamma = np.zeros((self.horizon*m*n, self.horizon*n))
        
    
        for i in range(horizon-1):
            Gamma[i*n*m:(i+1)*n*m, i*n:(i+1)*n] = -E
            Gamma[i*n*m:(i+1)*n*m, (i+1)*n:(i+2)*n] =  self.discount*P
        
        Gamma[ (horizon-1)*m*n:(horizon)*m*n , (horizon-1)*n: horizon*n] = -E
        
        left_eye = np.vstack([bold_I for _ in range(horizon)])
        
        Gamma  = np.hstack((left_eye,Gamma))
        
    #     return -np.min(np.linalg.svd(Gamma, compute_uv=False)[:-1])
        return Gamma

    def construct_Xi(self, pi):
        horizon, n_states, n_actions = pi.shape
    
        Xi = np.zeros((horizon*n_states*n_actions,1))
        
        for t in range(horizon):
            curr_pi = pi[t].flatten('F')[:,None]
    #         print(curr_pi.shape)
            # next_pi = pi[t+1].flatten('F')[:,None]
            Xi[t*n_states*n_actions:(t+1)*n_states*n_actions] = np.log(curr_pi) 
        return Xi    
    
    def prune_Gamma_and_Xi(self, pi):
        Reach, UnReach = self.get_reachable_tube()
        n_unreach_states = []
        total_unreach_states = set(UnReach[str(0)])

        Gamma = self.construct_Gamma()
        # print(Gamma.shape)
        Xi = self.construct_Xi(pi)

        remove_indices = []

        for t in range(self.horizon):
            UnReach_t = UnReach[str(t)]
            if t > 1:
                total_unreach_states = total_unreach_states.intersection(set(UnReach_t))
                n_unreach_states.append(len(total_unreach_states))
            # print(UnReach_t)
            for unreachable_state in UnReach_t:
                for a in range(self.n_actions):
                    remove_indices.append(  t*self.n_states*self.n_actions + (self.n_states*a + unreachable_state)) 

        # print(remove_indices)
        # print("len:", len(remove_indices))
        Gamma = np.delete(Gamma,obj = remove_indices, axis = 0)
        Xi = np.delete(Xi, obj = remove_indices, axis = 0 )                
        return Gamma, Xi , n_unreach_states


    def compute_projected_kernel(self,Gamma):
        K = scipy.linalg.null_space(Gamma)
        projected_K = K[:self.n_states*self.n_actions,:]
        # print("The shape of projeted matrix is: ", projected_K.shape)
        # print("The rank of projected K is: ", np.linalg.matrix_rank(projected_K))    
        return np.linalg.matrix_rank(projected_K)


class StripBlockingGridWorld(BasicGridWorld):
    def __init__(self, grid_size, wind, discount,horizon,start_state, feature_dim, p1, p2,theta, feature_type):
        super().__init__(grid_size, wind, discount,horizon,start_state, feature_dim, p1, p2,theta, feature_type)
    def __str__(self):
        return 'b'
    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
            
        
        # # is this state blocked in the MDP?
        # left_column = (i == 1) or (i == 2) or (i == 3) or (i == 4)
        # lc = [1,2,3,4]

        # disallow transition from the leftmost column to the one next to it
        if k in self.blocked_states:
            return 0.0

        # disallow self transitions for the left column
        # if i == k and left_column:
        #     return 0.0
        
        # # disallow transition from the leftmost column to the one next to it
        # if k in self.blocked_states and left_column:
        #     return 0.0

        # # disallow transition from the 2nd column to the left most column 
        # if k in lc and i in self.blocked_states:
        #     return 0.0
        


        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions


class WallBlockingGridWorld(BasicGridWorld):
    def __init__(self, grid_size, wind, discount,horizon,start_state, feature_dim, p1, p2,theta, feature_type):
        super().__init__(grid_size, wind, discount,horizon,start_state, feature_dim, p1, p2,theta, feature_type)

        self.make_state_determinstic(1)
        self.make_state_determinstic(2)
        self.make_state_determinstic(3)
        self.make_state_determinstic(4)

    def __str__(self):
        return 'c' # used for proper naming of the titles of the plots
        
    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
            
        
        # # is this state blocked in the MDP?
        left_column = (i == 1) or (i == 2) or (i == 3) or (i == 4)
        lc = [1,2,3,4]

        # disallow self transitions for the left column
        if i == k and left_column:
            return 0.0
        
        # disallow transition from the leftmost column to the one next to it
        if k in self.blocked_states and left_column:
            return 0.0

        # disallow transition from the 2nd column to the left most column 
        if k in lc and i in self.blocked_states:
            return 0.0
        


        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions