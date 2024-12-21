"""
Implementation of linear scalarization algorithm for Q-learning.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import numpy as np
import random
import os
from tqdm.rich import tqdm
import wandb
from algo.utils import WelfareFunc, log_if_wdb
from algo.vanilla_value_iteration import ValueIteration
import pdb

class LinearScalarize:
    def __init__(
        self, 
        env, 
        init_val, 
        episodes, 
        weights, 
        lr, 
        gamma, 
        epsilon, 
        welfare_func_name, 
        save_path, 
        nsw_lambda, 
        p=None, 
        seed=2023, 
        wdb=False,
        threshold=4,
        rho=0.4,
    ):
        self.env = env
        self.init_val = init_val
        self.welfare_func_name = welfare_func_name
        self.episodes = episodes
        self.weights = weights
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_path = save_path
        self.seed = seed
        self.wdb = wdb
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, rho)
        self.dim = len(self.env.loc_coords)
        if len(self.weights) != self.dim: 
            raise ValueError('Dimension of weights not same as dimension of rewards')      
        self.make_dir_path()
    
    def make_dir_path(self):
        assert hasattr(self, "save_path"), "save_path is not defined"
        dir_path = self.save_path.split("/")[0:-1]
        dir_path = "/".join(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory path is: {dir_path}")
    
    def greedy(self, vec):
        '''Helper function'''
        arr = []
        for val in vec: 
            arr.append(np.dot(self.weights, val))
        return np.argmax(arr)
    
    def initialize(self):
        self.Q = np.full([self.env.observation_space.n, self.env.action_space.n, self.dim], self.init_val, dtype=float)
        self.nonlinear_record = []
        self.Racc_record = []
    
    def train(self):
        self.initialize()
        for i in tqdm(range(1, self.episodes+1), desc=f"Training {self.welfare_func_name}..."):
            R_acc = np.zeros(self.dim)   # for recording performance, does not affect action selection
            state = self.env.reset()[0]
            done = False
            c = 0
            
            while not done:
                if np.random.uniform(0,1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.greedy(self.Q[state])
                    
                next, reward, _, done, info = self.env.step(action)
                next_action = self.greedy(self.Q[next])
                
                for j in range(len(self.Q[state, action])):
                    self.Q[state,action][j] = self.Q[state,action][j] + self.lr*(reward[j]+self.gamma*self.Q[next,next_action][j]-self.Q[state,action][j])
                    
                state = next
                R_acc += np.power(self.gamma, c)*reward
                c += 1
            
            R_acc = np.where(R_acc < 0, 0, R_acc)
            
            if self.welfare_func_name == "nash-welfare":
                nonlinear_score = self.welfare_func.nash_welfare(R_acc)
            elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas"]:
                nonlinear_score = self.welfare_func(R_acc)
            else:
                raise ValueError("Invalid welfare function name")
            
            self.Racc_record.append(R_acc)
            self.nonlinear_record.append(nonlinear_score)
            print(f"R_acc: {R_acc}, {self.welfare_func_name}: {nonlinear_score}")
            
            if self.wdb:
                wandb.log({self.welfare_func_name: nonlinear_score})
            
        print("Finish training")
        np.savez(self.save_path, Racc_record=np.asarray(self.Racc_record), nonlinear_record=np.asarray(self.nonlinear_record))
        print(f"Results saved at {self.save_path}")


class LinearScalarizeM(ValueIteration):
    def __init__(
        self, 
        env, 
        reward_dim, 
        gamma, 
        epsilon, 
        save_path, 
        welfare_func_name, 
        p, 
        episodes, 
        nsw_lambda, 
        lr, 
        weights, 
        seed=1122, 
        tol=1e-6, 
        log_every=100, 
        wdb=False,
        val_load_path=None,
        threshold=4,
        rho=0.4,
    ):
        super().__init__(env, reward_dim, gamma, seed, tol, log_every)
        self.episodes = episodes
        self.weights = weights
        self.lr = lr
        self.epsilon = epsilon
        self.welfare_func_name = welfare_func_name
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, rho)
        self.nsw_lambda = nsw_lambda
        self.p = p
        self.save_path = save_path
        self.wdb = wdb
        if val_load_path:
            self.load_val(val_load_path)
    
    def load_val(self, val_load_path):
        data = np.load(val_load_path)
        self.Q_arr = data["Q_arr"]

    def initialize(self):
        if not hasattr(self, "Q_arr"):
            self.value_iteration_over_d()
        self.Q = np.transpose(self.Q_arr, (1, 2, 0))
        self.Racc_record, self.nonlinear_record = [], []
        assert self.Q.shape == (self.env.observation_space.n, self.env.action_space.n, self.reward_dim), f"Q shape is {self.Q.shape}"
    
    def greedy(self, vec):
        '''Helper function'''
        arr = []
        for val in vec: 
            arr.append(np.dot(self.weights, val))
        return np.argmax(arr)
    
    def train(self):
        self.initialize()
        random.seed(self.seed)
        np.random.seed(self.seed)
        state = self.env.reset(seed=self.seed)
        state = state[0]
        print(f"Initial state for evaluation: {state}")
        Racc = np.zeros(self.reward_dim)
        c = 0
        done = False

        while not done:
            action = self.greedy(self.Q[state])
                
            next, reward, _, done, info = self.env.step(action)        
            state = next
            Racc += np.power(self.gamma, c)*reward
            c += 1
        
        log_if_wdb({"welfare_func": self.welfare_func_name})

        if self.welfare_func_name == "nash-welfare":
            nonlinear_score = self.welfare_func.nash_welfare(Racc)
        elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas"]:
            nonlinear_score = self.welfare_func(Racc)
        
        log_if_wdb({"welfare_val": nonlinear_score})
        print(f"welfare_val: {nonlinear_score}, Racc: {Racc}")