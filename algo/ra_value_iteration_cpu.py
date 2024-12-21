"""Implemtation of RAVI on CPU (with parallelization)"""
import concurrent.futures
import itertools
import sys
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from sys import getsizeof
from time import perf_counter

import numpy as np
import wandb
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from algo.utils import DiscreFunc, WelfareFunc, log_if_wdb

sys.path.insert(0, '../envs')
import multiprocessing as mp
import os
import time
import warnings
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import envs

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

def calc_vals_mp(Racc_code, Racc, transitions, rewards, nexts, t, gamma, time_horizon, discre_func, encode_Racc_dict, n_actions, n_states, reward_dim, V):
    # calculate value function for each state, vectorized over states to reduce computation time
    next_Racc = Racc + np.power(gamma, time_horizon - t) * rewards
    next_Racc_discretized = discre_func(next_Racc).reshape((-1, reward_dim))
    next_Racc_discretized = [r for r in next_Racc_discretized]
    next_Racc_code = np.array([encode_Racc_dict[tuple(r)] for r in next_Racc_discretized])
    V = V[nexts.astype(int).flatten(), next_Racc_code, t - 1].reshape((n_states, n_actions))
    all_V = transitions * V
    return np.max(all_V, axis=1), np.argmax(all_V, axis=1), Racc_code


class RAValueIterationCPU:
    def __init__(
        self, 
        env, 
        discre_alpha, 
        growth_rate, 
        gamma, 
        reward_dim, 
        time_horizon, 
        welfare_func_name, 
        nsw_lambda, 
        save_path, 
        seed=1122, 
        p=None, 
        threshold=5, 
        wdb=False, 
        scaling_factor=1,
        parallel=True,
        multi_threading=True,
        multi_processing=False,
        eval = False,
        load_path = None,
        rho = 0.4,
    ):
        self.env = env
        self.welfare_func_name = welfare_func_name
        self.discre_alpha = discre_alpha # Discretization factor for accumulated rewards.
        self.discre_func = DiscreFunc(discre_alpha, growth_rate)  # Discretization function with growth rate for exponential discretization.
        self.gamma = gamma
        self.scaling_factor = scaling_factor
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, alpha=rho)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
        self.save_path = save_path
        self.make_dir_path()
        self.parallel = parallel
        self.multi_threading = multi_threading
        self.multi_processing = multi_processing
        self.eval = eval
        self.load_path = load_path
        if self.eval:
            assert self.load_path is not None, "need to have load_path if in eval = True"

        if self.parallel:
            assert self.multi_threading or self.multi_processing, "Must specify either multi-threading or multi-processing"
            if self.multi_threading:
                assert not self.multi_processing, "Cannot use both multi-threading and multi-processing"
                print("Using multi-threading")
            elif self.multi_processing:
                print("Using multi-processing")
        else:
            print("Using serial computation")
    
    def make_dir_path(self):
        assert hasattr(self, "save_path"), "save_path is not defined"
        dir_path = self.save_path.split("/")[0:-1]
        dir_path = "/".join(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory path is: {dir_path}")
    
    def initialize(self):
        # Calculate maximum possible reward accumulation for exponential discretization
        if self.gamma == 1:  # Special case where gamma is 1, just multiply max reward by the number of steps
            max_reward = self.time_horizon/self.scaling_factor
        else:
            # Calculate the sum of the geometric series
            sum_of_discounts = (1 - self.gamma ** ((self.time_horizon+1)/self.scaling_factor)) / (1 - self.gamma)
            max_reward = sum_of_discounts
        max_discrete = self.discre_func(max_reward)

        print(f"Max discretized reward during initialization: {max_discrete}")

        alpha_arr = []
        for alpha in tqdm(range(0, int(np.ceil(max_discrete / self.discre_alpha)) + 1), desc=f"Initialize discretization grid with scaling factor of {self.scaling_factor}..."):
            alpha_arr.append(self.discre_func(alpha * self.discre_alpha))
        self.discre_grid = np.unique(alpha_arr)


        init_Racc = []
        for r in tqdm(list(itertools.product(self.discre_grid, repeat=self.reward_dim)), desc="Initialize possible reward accumulations..."):
            init_Racc.append(np.asarray(r))
        self.init_Racc = init_Racc

        encode_Racc_dict = {}
        for i, r in enumerate(tqdm(self.init_Racc, desc="Initialize reward accumulation encoding...")):
            encode_Racc_dict[tuple(r)] = i
        self.encode_Racc_dict = encode_Racc_dict
        
        print("Initializing V function...")
        self.V = np.zeros((
            self.env.observation_space.n, 
            int(len(self.discre_grid) ** self.reward_dim),
            self.time_horizon + 1, 
        ), dtype=np.float32)
        memory, units = self.memory_usage(self.V)
        print(f"V function memory: {memory:.2f} {units}, shape: {self.V.shape}")
        print("Initializing Pi function...")
        self.Pi = self.V.copy()
        print("Finish initializing Pi and V functions")
        
        for Racc in tqdm(self.init_Racc, desc="Initializing V function with W(Racc)..."):
            Racc_code = self.encode_Racc(Racc)
            self.V[:, Racc_code, 0] = self.welfare_func(Racc)

        if self.parallel:
            print(f"\nThere are {os.cpu_count()} CPUs available")
    
    def memory_usage(self, vars):
        if isinstance(vars, list):
            memory_usage = 0    
            for e in vars:
                memory_usage += getsizeof(e)
        else:
            memory_usage = getsizeof(vars)
        
        geq_three, counter = True, 0
        while geq_three:
            geq_three = memory_usage >= 1000
            if geq_three:
                memory_usage /= 1024
                counter += 1

        units = ['B', 'KB', 'MB', 'GB', 'TB']
        return memory_usage, units[counter]
    
    def encode_Racc(self, Racc):
        # Encode the accumulated reward for indexing.
        assert hasattr(self, "encode_Racc_dict"), "need to have initialize accumulated reward to begin with"
        return self.encode_Racc_dict[tuple(Racc)]

    def iterate_Racc_serial(self, curr_Racc, t):
        for Racc in curr_Racc:
            Racc_code = self.encode_Racc(Racc)
            
            for state in range(self.env.observation_space.n):
                transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)        # vectorized, in dimensionality of the action space
                
                next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * reward_arr     # use broadcasting
                next_Racc_discretized = self.discre_func(next_Racc) # Discretize next rewards.
                next_Racc_code = [self.encode_Racc(d) for d in next_Racc_discretized]
                
                all_V = transition_prob * self.V[next_state_arr.astype(int), next_Racc_code, t - 1]
                self.V[state, Racc_code, t] = np.max(all_V)
                self.Pi[state, Racc_code, t] = np.argmax(all_V)
    
    def iterate_Racc_parallel(self, curr_Racc, t):
        # prepare variables for multiprocessing
        Racc_codes_arr = np.asarray([self.encode_Racc(r) for r in curr_Racc])
        curr_Racc_arr = np.array(curr_Racc, dtype=np.float32)
        transition_arr = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=np.float32)
        reward_arr = np.zeros((self.env.observation_space.n, self.env.action_space.n, self.reward_dim), dtype=np.float32)
        next_arr = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=np.int32)

        # populate transition, reward, and next state arrays, do this outside the loop to avoid repeated calls
        for state in range(self.env.observation_space.n):
            transition_arr[state], reward_arr[state], next_arr[state] = self.env.get_transition(state)
        
        if self.multi_threading:
            def calc_vals_mt(Racc_code, Racc, transitions, rewards, nexts):
                # calculate value function for each state, vectorized over states to reduce computation time
                next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * rewards
                next_Racc_discretized = self.discre_func(next_Racc).reshape((-1, self.reward_dim))
                next_Racc_discretized = [r for r in next_Racc_discretized]
                next_Racc_code = np.array([self.encode_Racc(r) for r in next_Racc_discretized])
                V = self.V[nexts.astype(int).flatten(), next_Racc_code, t - 1].reshape((self.env.observation_space.n, self.env.action_space.n))
                all_V = transitions * V
                self.V[:, Racc_code, t] = np.max(all_V, axis=1)
                self.Pi[:, Racc_code, t] = np.argmax(all_V, axis=1)
            
            # multithreading, itertools.repeat returns an iterator that will return the same value indefinitely
            # this is used to pass the same values to each process
            with ThreadPoolExecutor() as executor:
                executor.map(
                    calc_vals_mt, 
                    Racc_codes_arr, 
                    curr_Racc_arr, 
                    itertools.repeat(transition_arr), 
                    itertools.repeat(reward_arr), 
                    itertools.repeat(next_arr),
                )
        
        elif self.multi_processing:
            sm = SharedMemory(size=self.V.nbytes, create=True)
            V_shared = np.ndarray(self.V.shape, dtype=np.float32, buffer=sm.buf)
            V_shared[:] = self.V[:]
        
            with ProcessPoolExecutor() as executor:
                results = executor.map(
                    calc_vals_mp, 
                    Racc_codes_arr,
                    curr_Racc_arr,
                    itertools.repeat(transition_arr), 
                    itertools.repeat(reward_arr), 
                    itertools.repeat(next_arr),
                    itertools.repeat(t),
                    itertools.repeat(self.gamma),
                    itertools.repeat(self.time_horizon),
                    itertools.repeat(self.discre_func),
                    itertools.repeat(self.encode_Racc_dict),
                    itertools.repeat(self.env.action_space.n),
                    itertools.repeat(self.env.observation_space.n),
                    itertools.repeat(self.reward_dim),
                    itertools.repeat(V_shared),
                )
                for res in results:
                    self.V[:, res[2], t] = res[0]
                    self.Pi[:, res[2], t] = res[1]

    def train(self):
        self.initialize()
        
        if not self.eval:
            mem_usage, units = self.memory_usage([
                self.V, self.Pi, self.encode_Racc_dict, self.init_Racc, self.discre_grid
            ])
            print(f"Memory usage: {mem_usage:.2f} {units}")
            log_if_wdb({f"memory usage {units}": mem_usage})
            
            for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
                # Use the discretized grid from initialization that accounts for exponential growth
                # Calculate the sum of the geometric series for the remaining timesteps
                remaining_steps = self.time_horizon - t
                if self.gamma == 1:
                    max_accumulated_discounted_reward = remaining_steps
                else:
                    max_accumulated_discounted_reward = (1 - self.gamma ** (remaining_steps)) / (1 - self.gamma)

                max_possible_reward = min(max_accumulated_discounted_reward, self.discre_grid[-2])
                max_possible_discretized_reward = self.discre_func(max_possible_reward)

                while max_possible_discretized_reward < 1:
                    max_possible_discretized_reward = self.discre_func(max_possible_discretized_reward+self.discre_alpha)

                print(f"\nMax discretized reward at t = {remaining_steps}: {max_possible_discretized_reward}")
                
                # Filter the discretization grid based on the computed max possible reward
                time_grid = self.discre_grid[self.discre_grid <= max_possible_discretized_reward]
                curr_Racc = [np.asarray(r) for r in list(itertools.product(time_grid, repeat=self.reward_dim))] # Current possible rewards.

                if self.parallel:
                    self.iterate_Racc_parallel(curr_Racc, t)
                else:
                    self.iterate_Racc_serial(curr_Racc, t)
        
            print("Finish training")
            np.savez(self.save_path, V=self.V, Pi=self.Pi)
            print(f"V and Pi functions are saved as {self.save_path}")
        else:
            del self.V
            del self.Pi
            data = np.load(self.load_path)
            self.V, self.Pi = data["V"], data["Pi"]

        self.evaluate(final=True)
    
    def evaluate(self, final=False):
        random.seed(self.seed)
        np.random.seed(self.seed)
        state = self.env.reset(seed=self.seed)
        state = state[0]
        print(f"Initial state for evaluation: {state}")

        Racc = np.zeros(self.reward_dim)
        c = 0
        
        for t in range(self.time_horizon, 0, -1):
            Racc_code = self.encode_Racc(self.discre_func(Racc))
            action = self.Pi[state, Racc_code, t]
            
            next, reward, done, truncated, info = self.env.step(action)
            state = next
            Racc += np.power(self.gamma, c) * reward
            print(f"Accumulated Reward at t = {self.time_horizon-t}: {Racc}")
            c += 1
        
        if self.welfare_func_name == "nash-welfare":
            log_if_wdb({"welfare_val": self.welfare_func.nash_welfare(Racc)})
            print(f"welfare_val: {self.welfare_func.nash_welfare(Racc)}, Racc: {Racc}")

        elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas"]:
            log_if_wdb({"welfare_val": self.welfare_func(Racc)})
            print(f"welfare_val: {self.welfare_func(Racc)}, Racc: {Racc}")
        
        log_if_wdb({"welfare_func": self.welfare_func_name})
        if final:
            self.Racc_record = Racc
        