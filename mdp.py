import numpy as np
from reward_machine.reward_machine import RewardMachine
import scipy.linalg

class MDP(object):
    def __init__(self, n_states, n_actions, P, gamma, horizon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.gamma = gamma
        self.horizon = horizon 


class MDPRM(object):
    def __init__(self, mdp : MDP, rm : RewardMachine, L : dict):
        self.mdp = mdp 
        self.rm = rm
        self.L = L
        
    def sua_pair_from_i(self,i):
        a = i // (self.mdp.n_states*self.rm.n_states)
        r = i % (self.mdp.n_states*self.rm.n_states)
        u = r // self.mdp.n_states
        s = r % self.mdp.n_states
        return (s,u,a)



    def su_pair_from_s(self,i):
       
        return  ( i % self.mdp.n_states ,  i  // self.mdp.n_states)  # (s,u) pair

    def s_from_su_pair(self, su):

        (s,u) = su
        return u*self.mdp.n_states + s

    def construct_product(self):
        n_mdp_states = self.mdp.n_states # number of mdp states
        n_rm_states = len(self.rm.U)     # number of RM states
        n_mdprm_states = n_mdp_states*n_rm_states # number of product states

        n_mdprm_actions = self.mdp.n_actions # number of product actions

        P_mdprm = []

        for a in range(n_mdprm_actions):
            P = np.zeros(shape = (n_mdprm_states,n_mdprm_states))
            for s in range(n_mdprm_states):
                for s_prime in range(n_mdprm_states):
                    (s1,u1) = self.su_pair_from_s(s)
                    (s1_prime,u1_prime) = self.su_pair_from_s(s_prime)

                    if u1_prime == self.rm._compute_next_state(u1, self.L[s1_prime]):
                        P[s,s_prime] = self.mdp.P[a][s1,s1_prime] 
                   
            P_mdprm.append(P)

        return MDP(n_states=n_mdprm_states, n_actions=n_mdprm_actions,P = P_mdprm,gamma = self.mdp.gamma,horizon=self.mdp.horizon)
   