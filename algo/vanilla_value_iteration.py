import numpy as np
from abc import ABC, abstractmethod
from tqdm.rich import tqdm

class ValueIteration(ABC):
    def __init__(
        self, 
        env, 
        reward_dim, 
        gamma, 
        seed=1122, 
        tol=1e-6, 
        log_every=100
    ):
        self.env = env
        self.gamma = gamma
        self.reward_dim = reward_dim
        self.seed = seed
        self.tol = tol
        self.V_arr = {d : np.zeros(self.env.observation_space.n) for d in range(self.reward_dim)}
        self.log_every = log_every
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

    def value_iteration_over_d(self):
        for d in range(self.reward_dim):
            print(f"*** Value iteration for reward dimension {d} ***")
            delta = float('inf')
            c = 0
            while delta >= self.tol:
                delta = 0
                for s in range(self.env.observation_space.n):
                    v = self.V_arr[d][s]
                    prob_arr, reward_arr, next_arr = self.env.get_transition(s)     # NOTE: taxi environment is deterministic, so we don't need to iterate over s' and r for p(s',r|s,a)
                    self.V_arr[d][s] = max([p * (r[d] + self.gamma * self.V_arr[d][int(s_)]) for p, s_, r in zip(prob_arr, next_arr, reward_arr)])
                    delta = max(delta, abs(v - self.V_arr[d][s]))
                    c += 1
                    if c % self.log_every == 0:
                        print(f"Value iteration: Iteration {c}, delta: {delta}, tolerance: {self.tol}")

        self.Q_arr = []
        for d in range(self.reward_dim):
            print(f"*** Constructing Q-table for reward dimension {d} ***")
            Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            for s in range(self.env.observation_space.n):
                prob_arr, reward_arr, next_arr = self.env.get_transition(s)
                for a, (p, s_, r) in enumerate(zip(prob_arr, next_arr, reward_arr)):
                    tmp = p * (r[d] + self.gamma * self.V_arr[d][int(s_)])
                    Q[s,a] = tmp
            self.Q_arr.append(Q)
            del self.V_arr[d]
        del self.V_arr
        print("====== Value iteration COMPLETED =====")
