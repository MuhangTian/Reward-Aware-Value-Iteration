"""
Implementation of Welfare Q-learning algorithm.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import os

import numpy as np
import wandb
from tqdm import tqdm

from algo.utils import WelfareFunc


class WelfareQ:
    def __init__(
        self, 
        env, 
        lr, 
        gamma, 
        epsilon, 
        episodes, 
        init_val, 
        welfare_func_name, 
        nsw_lambda, 
        save_path, 
        dim_factor, 
        p=None, 
        non_stationary=True, 
        seed=1122, 
        wdb=False,
        threshold=4,
        rho=0.4,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.welfare_func_name = welfare_func_name
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold=threshold, alpha=rho)
        self.p = p
        self.save_path = save_path
        self.seed = seed
        self.epsilon = epsilon
        self.wdb = wdb
        self.episodes = episodes
        self.init_val = init_val
        self.non_stationary = non_stationary
        self.nsw_lambda = nsw_lambda
        self.dim_factor = dim_factor
        self.make_dir_path()
    
    def make_dir_path(self):
        assert hasattr(self, "save_path"), "save_path is not defined"
        dir_path = self.save_path.split("/")[0:-1]
        dir_path = "/".join(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory path is: {dir_path}")
    
    def initialize(self):
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n, len(self.env.loc_coords)], dtype=float)
        self.Q = self.Q + self.init_val
        self.Racc_record = []
        self.nonlinear_record = []
    
    def argmax_egalitarian(self, R_acc, vec):
        '''Helper function for egalitarian welfare'''
        if np.all(R_acc == 0):
            idx = np.random.randint(0, len(R_acc))      # NOTE: if all elements are 0, then randomly select idx
        else:
            idx = np.argmin(R_acc)
        arr = []
        for val in vec: 
            arr.append(val[idx])
        return np.argmax(arr)
    
    def argmax_welf_func(self, vec):
        return np.argmax([self.welfare_func(v) for v in vec])
        
    def train(self):
        self.initialize()
        
        for i in range(1, self.episodes + 1):
            R_acc = np.zeros(len(self.env.loc_coords))
            state = self.env.reset()
            if not isinstance(state, int):
                try:
                    state = state[0]
                except:
                    raise TypeError("Check initial state!")
            done = False
            c = 0
        
            while not done:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    if self.non_stationary:
                        if self.welfare_func_name == "egalitarian":
                            action = self.argmax_egalitarian(R_acc, R_acc + np.power(self.gamma, c) * self.Q[state])
                        else:
                            action = self.argmax_welf_func(R_acc + np.power(self.gamma, c) * self.Q[state])
                    else:  # if stationary policy, then R_acc doesn't affect action selection
                        raise NotImplementedError("Stationary policy is not implemented yet")
                

                next, reward, _, done, info = self.env.step(action)
                if self.welfare_func_name == "egalitarian":
                    max_action = self.argmax_egalitarian(R_acc, self.gamma * self.Q[next])
                elif self.welfare_func_name in ["RD-threshold", "Cobb-Douglas"]:
                    max_action = self.argmax_welf_func(R_acc + self.gamma * self.Q[next])
                else:
                    max_action = self.argmax_welf_func(self.gamma * self.Q[next])
                    
                self.Q[state, action] = self.Q[state, action] + self.lr * (reward + self.gamma * self.Q[next, max_action] - self.Q[state, action])
                
                self.epsilon = max(0.1, self.epsilon - self.dim_factor)  # epsilon diminishes over time
                state = next
                R_acc += np.power(self.gamma, c) * reward
                c += 1
        
            R_acc = np.where(R_acc < 0, 0, R_acc)  # Replace the negatives with 0
            
            if self.welfare_func_name == "nash-welfare":
                nonlinear_score = self.welfare_func.nash_welfare(R_acc)
            elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas", "utilitarian"]:
                nonlinear_score = self.welfare_func(R_acc)
                
            self.Racc_record.append(R_acc)
            self.nonlinear_record.append(nonlinear_score)
            print(f"R_acc: {R_acc}, {self.welfare_func_name}: {nonlinear_score}")
            
            if self.wdb:
                wandb.log({self.welfare_func_name: nonlinear_score})
        
        print("Finish training")
        np.savez(self.save_path, Racc_record=np.asarray(self.Racc_record), nonlinear_record=np.asarray(self.nonlinear_record))
        print(f"Results saved at {self.save_path}")

