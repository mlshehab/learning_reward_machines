import numpy as np
import gym
import time 

from scipy.special import softmax, logsumexp


class DisDyn(object):

    def __init__(self, env, bins):

        self.env  = env
        self.bins = bins
        self.lb_v = -3.0
        self.ub_v = 3.0
        self.n_obs_states = self.env.observation_space.shape[0]
        self.lb       = self.env.observation_space.low
        self.lb[1]    = self.lb_v    # since the bounds on the velocities are defined to be inf
        self.lb[-1]   = self.lb_v 
        self.ub       = self.env.observation_space.high
        self.ub[1]    = self.ub_v
        self.ub[-1]   = self.ub_v 

        self.n_states  = self.bins**self.n_obs_states
        self.n_actions = self.env.action_space.n

        self.seen_dict = {}
        self.simulated_dict = {}

        self.state_to_tuple_dict = {}
        self.tuple_to_state_dict = {}
        self.d2c_d = {}
        
        self.future_states_dict = [{},{}]


    def tuple_to_state(self, t):
        if t in self.tuple_to_state_dict:
            return self.tuple_to_state_dict[t]
        
        else:
            s1,s2,s3,s4 = t
            self.tuple_to_state_dict[t] = s4 + s3*self.bins + s2*self.bins**2 + s1*self.bins**3

        return s4 + s3*self.bins + s2*self.bins**2 + s1*self.bins**3

    def state_to_tuple(self,s):
        if s in self.state_to_tuple_dict:
            return self.state_to_tuple_dict[s]
        else:
            s1 = s // self.bins**3
            rem = s % self.bins**3
            
            s2 = rem // self.bins**2
            
            rem2 = rem % self.bins**2
            
            s3 = rem2 // self.bins
            s4 = rem2 % self.bins

            self.state_to_tuple_dict[s] = (s1,s2,s3,s4)

        return (s1,s2,s3,s4)


    def discrete_to_continuous(self,s_d, i):
        # i is the observation number 
        bin_width = (self.ub[i] - self.lb[i])/self.bins
        
        return self.lb[i] + (s_d + 0.5) * bin_width
    
    def continuous_to_discrete(self, s_c, i):
        # i is the observation number
        bin_width = (self.ub[i] - self.lb[i])/self.bins
        if s_c <= self.lb[i]:
            return 0
        if s_c >= self.ub[i]:
            return self.bins - 1
        
        return int((s_c - self.lb[i]) // bin_width)   

  
    def state_to_tuple_then_d2c(self, state):
        if state in self.seen_dict:
            return self.seen_dict[state]
        else:
            (s1,s2,s3,s4) = self.state_to_tuple(state)

            # make them continuous
            state_c = (self.discrete_to_continuous(s1,0),\
                        self.discrete_to_continuous(s2,1),\
                        self.discrete_to_continuous(s3,2),\
                        self.discrete_to_continuous(s4,3))
            
            self.seen_dict[state] = state_c
            
        return state_c

    def c2d_then_tuple_to_state(self,next_state_c):
       
        next_state_tuple = (self.continuous_to_discrete(next_state_c[0],0) , \
                                        self.continuous_to_discrete(next_state_c[1],1) , \
                                        self.continuous_to_discrete(next_state_c[2],2) , \
                                        self.continuous_to_discrete(next_state_c[3],3) )
                    
        next_state = self.tuple_to_state(next_state_tuple)
       
        return next_state
    
    def sample_state(self, state):

        assert state <= self.n_states

        state_tuple = self.state_to_tuple(state)
        # i is the observation number 
        ct_state = []

        for i in range(self.n_obs_states):
            s_d = np.random.uniform()
            bin_width = (self.ub[i] - self.lb[i])/self.bins
            ct_state.append(self.lb[i] + (state_tuple[i] + s_d) * bin_width)

        return np.array(ct_state)

    def find_next_state_dist(self,state,action, samples = 10):
        

        if state in self.future_states_dict[action]:
            return self.future_states_dict[action][state]
        
            
        possible_next_states = []
        terminated_status = []
        rewards = []

        for i in range(samples):
            # Sample a continuous state
            ct_state = self.sample_state(state)
            
            # Find the continuous next state 
            next_state_c, reward , terminated = dd.simulate(ct_state, action)

            next_state = dd.c2d_then_tuple_to_state(next_state_c)
            possible_next_states.append(next_state)
            terminated_status.append(terminated)
            rewards.append(reward)

        next_states, idx,  counts = np.unique(possible_next_states,return_index = True, return_counts = True)
        final_terminal = [terminated_status[i] for i in idx]
        final_rewards = [rewards[i] for i in idx]

        self.future_states_dict[action][state] = next_states, counts/counts.sum(), final_terminal, final_rewards

        return next_states, counts/counts.sum(), final_terminal, final_rewards


    def simulate(self, state, action):

        x, x_dot, theta, theta_dot = state
        force = self.env.force_mag if action == 1 else -self.env.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)      

        temp = (
            force + self.env.polemass_length * np.square(theta_dot) * sintheta
        ) / self.env.total_mass
        thetaacc = (self.env.gravity * sintheta - costheta * temp) / (
            self.env.length
            * (4.0 / 3.0 - self.env.masspole * np.square(costheta) / self.env.total_mass)
        )
        xacc = temp - self.env.polemass_length * thetaacc * costheta / self.env.total_mass

        if self.env.kinematics_integrator == "euler":
            x = x + self.env.tau * x_dot
            x_dot = x_dot + self.env.tau * xacc
            theta = theta + self.env.tau * theta_dot
            theta_dot = theta_dot + self.env.tau * thetaacc

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        
        terminated = bool(
            x < -self.env.x_threshold
            or x > self.env.x_threshold
            or theta < -self.env.theta_threshold_radians
            or theta > self.env.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

      

        return  np.array((x, x_dot, theta, theta_dot), dtype=np.float64) , reward , terminated

    def test_policy(self,policy, bellman):
        if bellman == 'soft':
            state, _ = self.env.reset()
            done = False
            eps_len = 0

            while not done:
                
                state_tuple = (self.continuous_to_discrete(state[0],0) , \
                            self.continuous_to_discrete(state[1],1) , \
                            self.continuous_to_discrete(state[2],2) , \
                            self.continuous_to_discrete(state[3],3) )
                            
                # this is the deterministic next state
                state_disc = self.tuple_to_state(state_tuple)
                p = policy[state_disc]

                action = np.random.choice(np.array([0,1]), p = p)
                next_state, reward , done, _,_ = self.env.step(action)

                state = next_state
            
                eps_len +=1

            print(f"The {bellman} policy lasted for {eps_len} episodes.") 

        else:
            state, _ = self.env.reset()
            done = False
            eps_len = 0

            while not done:
                
                state_tuple = (self.continuous_to_discrete(state[0],0) , \
                            self.continuous_to_discrete(state[1],1) , \
                            self.continuous_to_discrete(state[2],2) , \
                            self.continuous_to_discrete(state[3],3) )
                            
                # this is the deterministic next state
                state_disc = self.tuple_to_state(state_tuple)
                action = policy[state_disc]

              
                next_state, reward , done, _,_ = self.env.step(action)

                state = next_state
            
                eps_len +=1

            print(f"The {bellman} policy lasted for {eps_len} episodes.") 

def infinite_horizon_soft_bellman_iteration(env,dd:DisDyn, tol = 1e-4, logging = True, log_iter = 5, policy_test_iter = 20):

    # dd = DisDyn(env,bins)
    print(f"The total number of states is {dd.n_states} and n_bins = {dd.bins}. \n\n")

    gamma = 0.99
    n_actions = dd.n_actions
    n_states = dd.n_states

    v_soft = np.zeros((n_states,1)) # value functions
    q_soft = np.zeros((n_states, n_actions))

    v_hard = np.zeros((n_states,1)) # value functions
    q_hard = np.zeros((n_states, n_actions)) # state action value functions

    delta = np.inf 

    converged = delta < tol

    it = 0
    total_time = 0.0

    while not converged:
        
        it+=1

        start_time = time.time()

        for state in range(n_states): 
            for action in range(n_actions):

                next_states , p , terminated , rewards = dd.find_next_state_dist(state,action)

                future_value_soft = 0.0
                future_value_hard = 0.0

                for i in range(len(next_states)):
                    future_value_soft += gamma*(1 - terminated[i])*p[i]*v_soft[next_states[i]]
                    future_value_hard += gamma*(1 - terminated[i])*p[i]*v_hard[next_states[i]]


                q_soft[state,action] = rewards[0] +  future_value_soft
                q_hard[state,action] = rewards[0] +  future_value_hard

        v_new_soft = logsumexp(q_soft,axis = 1)
        v_new_hard = np.max(q_hard,axis = 1)

        end_time = time.time()
        total_time += end_time - start_time

        if logging and not it%log_iter and it >1 :
            print(f"Total: {it} iterations -- iter time: {end_time - start_time:.2f} sec -- Total time: {total_time:.2f}")
            print(f"Soft Error Norm ||e||: {np.linalg.norm(v_new_soft -v_soft):.6f} -- Hard Error Norm ||e||: {np.linalg.norm(v_new_hard -v_hard):.6f} ")
        
        # if logging and not it%policy_test_iter and it > 1:
        #     dd.test_policy(softmax(q_soft,axis = 1), bellman = 'soft')
        #     dd.test_policy(np.argmax(q_hard, axis = 1), bellman = 'hard')

        converged = np.linalg.norm(v_new_soft - v_soft) <= tol

        v_soft = v_new_soft
        v_hard = v_new_hard

    # find the policy
    soft_policy  = softmax(q_soft,axis = 1)
    hard_policy = softmax(q_hard,axis = 1)

    return q_soft,v_soft , soft_policy, q_hard , v_hard , hard_policy


env = gym.make('CartPole-v1')
dd = DisDyn(env, bins = 15)
print(f"The number of states is {dd.n_states}")
q_soft,v_soft , soft_policy, q_hard , v_hard , hard_policy = infinite_horizon_soft_bellman_iteration(env,dd)

for n_episodes in range(50):

    state, _ = env.reset()
    done = False
    eps_len = 0

    while not done:
        
        state_tuple = (dd.continuous_to_discrete(state[0],0) , \
                    dd.continuous_to_discrete(state[1],1) , \
                    dd.continuous_to_discrete(state[2],2) , \
                    dd.continuous_to_discrete(state[3],3) )
                    
        # this is the deterministic next state
        state_disc = dd.tuple_to_state(state_tuple)
        p = soft_policy[state_disc]

        action = np.random.choice(np.array([0,1]), p = p)
        next_state, reward , done, _,_ = env.step(action)

        state = next_state
       
        eps_len +=1

    print(f"The episode len is {eps_len}.")

env.close()