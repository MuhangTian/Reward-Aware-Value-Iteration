"""Implementation of RAVI with GPU parallelization"""
import itertools

import numpy as np
from tqdm.rich import tqdm
import sys
from sys import getsizeof
import warnings
from tqdm import TqdmExperimentalWarning

import wandb
from algo.utils import DiscreFunc, WelfareFunc, log_if_wdb

sys.path.insert(0, '../envs')
import envs

import os
import pdb
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class RAValueIterationGPU:
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
        rho=0.4,
    ):
        self.env = env
        self.welfare_func_name = welfare_func_name
        self.discre_alpha = discre_alpha # Discretization factor for accumulated rewards.
        self.discre_func = DiscreFunc(discre_alpha, growth_rate)  # Discretization function with growth rate for exponential discretization.
        self.gamma = gamma
        self.growth_rate = growth_rate
        self.scaling_factor = scaling_factor
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold, alpha=rho)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
        self.save_path = save_path

        self.num_actions = env.action_space.n  # Get the number of actions from the environment

        # Define CUDA kernel
        self.mod = SourceModule(f"""
#include <stdio.h>
#include <math.h>

__device__ float atomicMaxFloat(float* address, float val) {{
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;

    do {{
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }} while (assumed != old);

    return __int_as_float(old);
}}
                                
__device__ int atomicMaxInt(int* address, int val) {{
    int old = *address, assumed;
    do {{
        assumed = old;
        old = atomicCAS(address, assumed, max(val, assumed));
    }} while (assumed != old);
    return old;
}}

__device__ int discretize(float value, float alpha, float growth_rate) {{
    if (growth_rate > 1.0f) {{
        int index = roundf(log1pf(value / alpha * (growth_rate - 1.0f)) / logf(growth_rate));
        if (index < 0) {{
            // printf("Error: index < 0, value: %f, alpha: %f, growth_rate: %f, index: %d\\n", value, alpha, growth_rate, index);
            return 0;
        }}
        return index;
    }} else {{
        return roundf(value / alpha);
    }}
}}

__device__ int calculate_Racc_code(float *discre_grid, int grid_size, float *next_Racc, int reward_dim) {{
    int Racc_code = 0;
    int factor = 1;
    for (int i = reward_dim - 1; i >= 0; i--) {{
        int idx = -1;
        float min_diff = 1e6;
        for (int j = 0; j < grid_size; j++) {{
            float diff = fabs(next_Racc[i] - discre_grid[j]);
            // printf("Dim %d: next_Racc[i] = %f, discre_grid[j] = %f, diff = %f\\n", i, next_Racc[i], discre_grid[j], diff);
            if (diff < min_diff) {{
                min_diff = diff;
                idx = j;
            }}
        }}
        // printf("Dim %d: next_Racc[i] = %f, closest discre_grid value = %f, idx = %d\\n", i, next_Racc[i], discre_grid[idx], idx);
        Racc_code += idx * factor;
        factor *= grid_size;
    }}
    // printf("Computed Racc_code: %d\\n", Racc_code);
    return Racc_code;
}}

extern "C" __global__ void compute_values(float *V, int *Pi, float *transition_prob, float *reward_arr, int *next_state_arr,
                        float gamma, int time_horizon, float *curr_Racc, int *Racc_code, 
                        int num_states, int num_Racc_total, int num_Racc, int reward_dim, int num_actions, int t, float alpha, float growth_rate, float *discre_grid, int grid_size) {{
    extern __shared__ float shared_memory[];
    float *next_Racc = shared_memory;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states * num_Racc) return;

    int Racc_idx = idx / num_states;
    int state_idx = idx % num_states;

    if (Racc_idx >= num_Racc || state_idx >= num_states) {{
        // printf("Error: Racc_idx (%d) or state_idx (%d) out of bounds (num_Racc=%d, num_states=%d)\\n", Racc_idx, state_idx, num_Racc, num_states);
        return;
    }}

    int state = state_idx;
    int Racc_code_idx = Racc_code[Racc_idx];

    int V_idx = state * num_Racc_total * (time_horizon + 1) + Racc_code_idx * (time_horizon + 1) + t;

    // Ensure V_idx is within bounds
    if (V_idx >= num_states * num_Racc_total * (time_horizon + 1)) {{
        // printf("Error: V_idx out of bounds. V_idx=%d, Max=%d\\n", V_idx, num_states * num_Racc_total * (time_horizon + 1));
        return;
    }}

    float max_V = -INFINITY;
    int best_action = 0;

    // if (state == 5) {{
    //     printf("State: %d, Racc: [", state);
    //     for (int i = 0; i < reward_dim; i++) {{
    //         printf("%f", curr_Racc[Racc_idx * reward_dim + i]);
    //         if (i < reward_dim - 1) {{
    //             printf(", ");
    //         }}
    //     }}
    //     printf("], Racc_code: %d\\n", Racc_code_idx);

        for (int a = 0; a < num_actions; a++) {{
            float transition_probability = transition_prob[state * num_actions + a];
            int next_state = next_state_arr[state * num_actions + a];

            for (int r = 0; r < reward_dim; r++) {{
                next_Racc[threadIdx.x * reward_dim + r] = curr_Racc[Racc_idx * reward_dim + r] + powf(gamma, time_horizon - t) * reward_arr[(state * num_actions + a) * reward_dim + r];
            }}

            // printf("  action: %d, transition_probability: %f, next_state: %d\\n", a, transition_probability, next_state);
            // printf("  Next_Racc: [");
            // for (int r = 0; r < reward_dim; r++) {{
            //     printf("%f", next_Racc[threadIdx.x * reward_dim + r]);
            //     if (r < reward_dim - 1) {{
            //         printf(", ");
            //     }}
            // }}
            // printf("]\\n");

            int next_Racc_code = calculate_Racc_code(discre_grid, grid_size, next_Racc + threadIdx.x * reward_dim, reward_dim);

            // printf("  Next_Racc_code: %d\\n", next_Racc_code);

            float all_V = 0.0;
            
            int next_idx = next_state * num_Racc_total * (time_horizon + 1) + next_Racc_code * (time_horizon + 1) + (t - 1);
            // Ensure next_idx is within bounds
            if (next_idx >= num_states * num_Racc_total * (time_horizon + 1)) {{
                // printf("Error: next_idx out of bounds. next_idx=%d, Max=%d\\n", next_idx, num_states * num_Racc_total * (time_horizon + 1));
                return;
            }}
            all_V += transition_probability * V[next_idx];
            // printf("  Accessing V[%d, %d, %d] (flattened idx: %d) = %f\\n", next_state, next_Racc_code, t - 1, next_idx, V[next_idx]);

            if (all_V > max_V) {{
                max_V = all_V;
                best_action = a;
            }}
        }}

        // printf("  max_V: %f, best_action: %d\\n", max_V, best_action);
        // printf("  Updating V[%d] (flattened idx: %d) with %f\\n", V_idx, V_idx, max_V);
        // printf("  Updating Pi[%d] (flattened idx: %d) with %d\\n", V_idx, V_idx, best_action);
    // }}

    atomicMaxFloat(&V[V_idx], max_V);
    atomicMaxInt(&Pi[V_idx], best_action);
}}
""")
    
    def __initialize(self):
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
        for alpha in tqdm(range(0, int(np.ceil(max_discrete / self.discre_alpha)) + 1), desc="Initialize discretization grid..."):
            alpha_arr.append(self.discre_func(alpha * self.discre_alpha))
        self.discre_grid = np.unique(alpha_arr)

        init_Racc = []
        for r in tqdm(list(itertools.product(self.discre_grid, repeat=self.reward_dim)), desc="Initialize possible reward accumulations..."):
            init_Racc.append(np.asarray(r))
        self.init_Racc = init_Racc

        encode_Racc_dict = {}
        for i, r in enumerate(tqdm(self.init_Racc, desc="Initialize reward accumulation encoding...")):
            encode_Racc_dict[tuple(r)] = i      # NOTE: changed to tuple because string for hashing is slow
        self.encode_Racc_dict = encode_Racc_dict

        self.flat_encode_Racc_dict, self.flat_encode_Racc_str_lens = self.create_flat_encode_dict()
        
        print("Initializing V function...")
        self.V = np.zeros((
            self.env.observation_space.n, 
            int(len(self.discre_grid) ** self.reward_dim),
            self.time_horizon + 1, 
        ))
        memory, units = self.__memory_usage(self.V)
        print(f"V function memory: {memory:.2f} {units}, shape: {self.V.shape}")
        print("Initializing Pi function...")
        self.Pi = self.V.copy()
        print("Finish initializing Pi and V functions")
        
        for Racc in tqdm(self.init_Racc, desc="Initializing V function with W(Racc)..."):
            Racc_code = self.encode_Racc(Racc)
            self.V[:, Racc_code, 0] = self.welfare_func(Racc)
    
    def __memory_usage(self, vars):
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
    
    def create_flat_encode_dict(self):
        max_index = max(self.encode_Racc_dict.values())
        flat_dict = np.full((max_index + 1, 256), -1, dtype=np.int32)  # Initialize with -1 to detect invalid accesses
        str_lens = np.zeros(max_index + 1, dtype=np.int32)  # Lengths of strings for each index

        for k, v in tqdm(self.encode_Racc_dict.items(), desc="Flattening encode dict..."):
            for i, char in enumerate(str(k)):
                flat_dict[v, i] = ord(char)  # Convert char to ASCII int
            str_lens[v] = len(k)

        return flat_dict.flatten(), str_lens

    # Function to launch CUDA kernel
    def parallel_compute(self, V, Pi, transition_prob, reward_arr, next_state_arr, gamma, time_horizon, curr_Racc, Racc_code, num_states, num_Racc_total, num_Racc, reward_dim, num_actions, t, alpha, growth_rate, discre_grid, grid_size):
        V_flat = V.flatten().astype(np.float32)
        Pi_flat = Pi.flatten().astype(np.int32)  # Ensure Pi is treated as an integer array

        V_gpu = cuda.mem_alloc(V_flat.nbytes)
        Pi_gpu = cuda.mem_alloc(Pi_flat.nbytes)
        transition_prob_gpu = cuda.mem_alloc(transition_prob.nbytes)
        reward_arr_gpu = cuda.mem_alloc(reward_arr.nbytes)
        next_state_arr_gpu = cuda.mem_alloc(next_state_arr.nbytes)
        curr_Racc_gpu = cuda.mem_alloc(curr_Racc.nbytes)
        Racc_code_gpu = cuda.mem_alloc(Racc_code.nbytes)
        discre_grid_gpu = cuda.mem_alloc(discre_grid.nbytes)
        
        cuda.memcpy_htod(V_gpu, V_flat)
        cuda.memcpy_htod(Pi_gpu, Pi_flat)
        cuda.memcpy_htod(transition_prob_gpu, transition_prob)
        cuda.memcpy_htod(reward_arr_gpu, reward_arr)
        cuda.memcpy_htod(next_state_arr_gpu, next_state_arr)
        cuda.memcpy_htod(curr_Racc_gpu, curr_Racc)
        cuda.memcpy_htod(Racc_code_gpu, Racc_code)
        cuda.memcpy_htod(discre_grid_gpu, discre_grid)
        
        block_size = 256
        grid_size = (num_states * num_Racc + block_size - 1) // block_size
        shared_mem_size = block_size * (reward_dim * np.dtype(np.float32).itemsize)  # Size of shared memory per thread

        func = self.mod.get_function("compute_values")
        func(V_gpu, Pi_gpu, transition_prob_gpu, reward_arr_gpu, next_state_arr_gpu, np.float32(gamma), np.int32(time_horizon), curr_Racc_gpu, Racc_code_gpu, np.int32(num_states), np.int32(num_Racc_total), np.int32(num_Racc), np.int32(reward_dim), np.int32(num_actions), np.int32(t), np.float32(alpha), np.float32(growth_rate), discre_grid_gpu, np.int32(len(discre_grid)), block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_mem_size)
        
        cuda.memcpy_dtoh(V_flat, V_gpu)
        cuda.memcpy_dtoh(Pi_flat, Pi_gpu)
        
        V[:] = V_flat.reshape(V.shape)
        Pi[:] = Pi_flat.reshape(Pi.shape)
        
        V_gpu.free()
        Pi_gpu.free()
        transition_prob_gpu.free()
        reward_arr_gpu.free()
        next_state_arr_gpu.free()
        curr_Racc_gpu.free()
        Racc_code_gpu.free()
        discre_grid_gpu.free()
        
    def train(self):
        self.__initialize()

        # Pre-compute transitions, rewards, and next states for all states and actions
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        transition_prob = np.zeros((num_states, num_actions), dtype=np.float32)
        reward_arr = np.zeros((num_states, num_actions, self.reward_dim), dtype=np.float32)
        next_state_arr = np.zeros((num_states, num_actions), dtype=np.int32)

        for state in range(num_states):
            transition_prob[state], reward_arr[state], next_state_arr[state] = self.env.get_transition(state)
        
        # Flatten arrays for GPU transfer
        transition_prob_flat = transition_prob.flatten()
        reward_arr_flat = reward_arr.flatten()
        next_state_arr_flat = next_state_arr.flatten()

        num_Racc_total = self.V.shape[1]  # Get the total number of accumulated rewards

        print("Estimating memory...")
        memory_usage = self.__memory_usage([
            self.V, self.Pi, transition_prob_flat, reward_arr_flat, next_state_arr_flat, 
            self.gamma, self.time_horizon, num_states, num_Racc_total, self.reward_dim, 
            self.num_actions, self.discre_alpha, self.growth_rate, self.discre_grid.astype(np.float32),
        ])
        print(f"Memory usage: {memory_usage[0]:.2f} {memory_usage[1]}")
        log_if_wdb({f"memory usage {memory_usage[1]}": memory_usage[0]})
        
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

            while t != self.time_horizon and max_possible_discretized_reward < 1:
                max_possible_discretized_reward = self.discre_func(max_possible_discretized_reward+self.discre_alpha)

            print(f"\nMax discretized reward at t = {remaining_steps}: {max_possible_discretized_reward}")
            log_if_wdb({"max_discretized_reward": max_possible_discretized_reward, "t": remaining_steps})
            
            # Filter the discretization grid based on the computed max possible reward
            time_grid = self.discre_grid[self.discre_grid <= max_possible_discretized_reward]
            curr_Racc = [np.asarray(r) for r in list(itertools.product(time_grid, repeat=self.reward_dim))] # Current possible rewards.

            num_Racc = len(curr_Racc)
            curr_Racc_np = np.array(curr_Racc, dtype=np.float32).flatten()
            Racc_code = np.array([self.encode_Racc(r) for r in curr_Racc], dtype=np.int32)
            
            self.parallel_compute(
                self.V, self.Pi, transition_prob_flat, reward_arr_flat, next_state_arr_flat, 
                self.gamma, self.time_horizon, curr_Racc_np, Racc_code.flatten(), 
                num_states, num_Racc_total, num_Racc, self.reward_dim, self.num_actions, t,
                self.discre_alpha, self.growth_rate, self.discre_grid.astype(np.float32),
                len(self.discre_grid)
            )
    
        self.evaluate(final=True)
    
    def evaluate(self, final=False):
        # self.env.seed(self.seed)
        state = self.env.reset(seed=self.seed)
        state = state[0]
        # Ensure the renders directory exists within the specified save path
        renders_path = self.save_path + '/renders'
        os.makedirs(renders_path, exist_ok=True)
        img_path = self.save_path + f'/renders/env_render_init.png'
        if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
            self.env.render(save_path=img_path)
        Racc = np.zeros(self.reward_dim)
        c = 0
        
        for t in range(self.time_horizon, 0, -1):
            Racc_code = self.encode_Racc(self.discre_func(Racc))
            action = self.Pi[state, Racc_code, t]
            
            next, reward, done, truncated, info = self.env.step(action)
            if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
                img_path = self.save_path + f'/renders/env_render_{self.time_horizon-t}.png'
                self.env.render(save_path=img_path)
                # decoded_pos, decoded_status = self.env.decode_state(next)
                # print(f"Decoded Position: {decoded_pos}, Decoded Resource Status: {decoded_status}")
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
        