"""
Implementation of mixture policy.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import numpy as np
import os
from algo.utils import WelfareFunc, log_if_wdb
from algo.vanilla_value_iteration import ValueIteration
import wandb
from tqdm.rich import tqdm
import random

class MixturePolicy:
    def __init__(
        self, 
        env, 
        episodes, 
        time_horizon, 
        lr, 
        epsilon, 
        gamma, 
        init_val, 
        weights, 
        interval, 
        welfare_func_name, 
        save_path, 
        nsw_lambda, 
        p=None, 
        seed=1122, 
        wdb=False,
        threshold=4,
        rho=0.4,
    ):
        self.env = env
        self.episodes = episodes
        self.welfare_func_name = welfare_func_name
        self.time_horizon = time_horizon
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_val = init_val
        self.weights = weights
        self.interval = interval
        self.save_path = save_path
        self.make_dir_path()
        self.seed = seed
        self.wdb = wdb
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, rho)
        self.dim = len(self.env.loc_coords)
    
    def make_dir_path(self):
        assert hasattr(self, "save_path"), "save_path is not defined"
        dir_path = self.save_path.split("/")[0:-1]
        dir_path = "/".join(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory path is: {dir_path}")
        
    def greedy(self, vec, weights):
        '''Helper function'''
        arr = []
        for val in vec: 
            arr.append(np.dot(weights, val))    # linear scalarization
        return np.argmax(arr)
    
    def initialize(self):
        self.Racc_record = []
        self.dims = [i for i in range(len(self.weights))]
        self.policies = []
        self.nonlinear_record = []
        for dim in self.dims:    # Obtain set of policies
            q = np.full([self.env.observation_space.n, self.env.action_space.n, self.dim], self.init_val, dtype=float)
            self.policies.append(q)
        
    def train(self):
        self.initialize()
        for i in tqdm(range(1, self.episodes+1), desc=f"Training {self.welfare_func_name}..."):
            R_acc = np.zeros(self.dim)
            state = self.env.reset()[0]
            done = False
            count, dim, c = 0, 0, 0
            Q = self.policies[dim]
            weights = self.weights[dim]
            
            while not done:
                if count > int(self.time_horizon/self.dim/self.interval):   # determines the period of changing policies
                    dim += 1
                    if dim >= self.dim: 
                        dim = 0  # back to first objective after a "cycle"
                    Q = self.policies[dim]
                    weights = self.weights[dim]
                    count = 0   # change policy after t/d timesteps
                    
                if np.random.uniform(0, 1) < self.epsilon: 
                    action = self.env.action_space.sample()
                else: 
                    action = self.greedy(Q[state], weights)
                
                next, reward, _, done, info = self.env.step(action)
                count += 1
                next_action = self.greedy(Q[next], weights)
                
                for j in range(len(Q[state, action])):
                    Q[state,action][j] = Q[state,action][j] + self.lr*(reward[j]+self.gamma*Q[next,next_action][j]-Q[state,action][j])

                state = next
                R_acc += np.power(self.gamma, c)*reward
                c += 1
            
            R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
            
            if self.welfare_func_name == "nash-welfare":
                nonlinear_score = self.welfare_func.nash_welfare(R_acc)
            elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD_threshold", "Cobb-Douglas"]:
                nonlinear_score = self.welfare_func(R_acc)
            else:
                raise ValueError("Invalid welfare function name")
            
            self.Racc_record.append(R_acc)
            self.nonlinear_record.append(nonlinear_score)

            print(f"R_acc: {R_acc}, {self.welfare_func_name}: {nonlinear_score}")
            
            if self.wdb:
                wandb.log({self.welfare_func_name: nonlinear_score})
        
        print("Finish training")
        np.savez(self.save_path, policies=self.policies, Racc_record=np.asarray(self.Racc_record), nonlinear_record=np.asarray(self.nonlinear_record))
        print(f"Results saved at {self.save_path}") 


class MixturePolicyM(ValueIteration):
    def __init__(
        self, 
        env,
        reward_dim, 
        time_horizon,  
        gamma, 
        interval, 
        welfare_func_name, 
        nsw_lambda, 
        val_load_path,
        p=None, 
        seed=2023, 
        wdb=False, 
        tol=1e-6, 
        log_every=100,
        threshold=4,
        rho=0.4,
    ):
        super().__init__(env, reward_dim, gamma, seed, tol, log_every)

        self.time_horizon = time_horizon
        self.interval = interval
        self.welfare_func_name = welfare_func_name
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, rho)
        self.wdb = wdb
        if val_load_path:
            self.load_val(val_load_path)
    
    def load_val(self, val_load_path):
        data = np.load(val_load_path)
        self.Q_arr = data["Q_arr"]

    def initialize(self):
        if not hasattr(self, "Q_arr"):
            self.value_iteration_over_d()
        self.Q = np.transpose(self.Q_arr, (1,2,0))
        self.Racc_record, self.nonlinear_record = [], []
        assert self.Q.shape == (self.env.observation_space.n, self.env.action_space.n, self.reward_dim), f"Q shape is {self.Q.shape}"
    
    def train(self):
        self.initialize()
        random.seed(self.seed)
        np.random.seed(self.seed)
        state = self.env.reset(seed=self.seed)
        state = state[0]
        print(f"Initial state for evaluation: {state}")
        Racc = np.zeros(self.reward_dim)
        c, count, dim = 0, 0, 0
        done = False

        while not done:
            if count > int(self.time_horizon/self.reward_dim/self.interval):   # determines the period of changing policies
                dim += 1
                if dim >= self.reward_dim: 
                    dim = 0
                count = 0   # change policy after t/d timesteps

            action = np.argmax(self.Q[state, :, dim])
                
            next, reward, _, done, info = self.env.step(action)        
            state = next
            Racc += np.power(self.gamma, c)*reward
            c += 1
            count += 1
        
        log_if_wdb({"welfare_func": self.welfare_func_name})

        if self.welfare_func_name == "nash-welfare":
            nonlinear_score = self.welfare_func.nash_welfare(Racc)
        elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas"]:
            nonlinear_score = self.welfare_func(Racc)
        
        log_if_wdb({"welfare_val": nonlinear_score})
        print(f"welfare_val: {nonlinear_score}, Racc: {Racc}")
    
