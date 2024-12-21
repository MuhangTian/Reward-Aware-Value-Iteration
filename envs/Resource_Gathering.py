import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ResourceGatheringEnv(gym.Env):
    def __init__(self, time_horizon, grid_size=(5, 5), num_resources=6, seed=None):
        super(ResourceGatheringEnv, self).__init__()
        self.grid_size = grid_size
        self.num_resources = num_resources

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Total number of possible states
        self.total_states = (grid_size[0] * grid_size[1]) * (2 ** num_resources)
        
        # Observation space: Discrete space of all possible states
        self.observation_space = spaces.Discrete(self.total_states)

        # Generate fixed resource and enemy positions at initialization
        self.generate_positions(seed)

        self.loc_coords = [[0,0],[3,2]] # Placeholder for compatibility
        self.timesteps = 0
        self.time_horizon = time_horizon

        self.reset(seed)

    def generate_positions(self, seed=None):
        super().reset(seed=seed)

        # Generate unique resource positions
        self.resource_positions = []
        while len(self.resource_positions) < self.num_resources:
            position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
            if position not in self.resource_positions:
                self.resource_positions.append(position)
        
        # Generate unique enemy positions
        self.enemy_positions = []
        while len(self.enemy_positions) < self.grid_size[0] * self.grid_size[1] // 3:
            position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
            if position not in self.enemy_positions and position not in self.resource_positions:
                self.enemy_positions.append(position)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.timesteps = 0
        # Initialize agent position and ensure it is not on a resource or enemy position
        while True:
            self.agent_pos = [np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])]
            if tuple(self.agent_pos) not in self.resource_positions and tuple(self.agent_pos) not in self.enemy_positions:
                break

        # print(f"Agent spawning location: {self.agent_pos}")
        
        # Initialize resource status (1 = present, 0 = collected)
        self.resources_status = np.ones(self.num_resources, dtype=np.int32)

        self.collected_resources = 0
        self.damage_taken = 0

        return self._get_obs()

    def _get_obs(self):
        # Flatten agent position
        agent_position_flat = self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]

        # Convert resource status to a single integer (binary representation)
        resource_status_int = int(''.join(map(str, self.resources_status)), 2)

        # Combine agent position and resource status into a single state
        obs = agent_position_flat * (2 ** self.num_resources) + resource_status_int

        return obs

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0: # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size[0] - 1: # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size[1] - 1: # right
            self.agent_pos[1] += 1

        self.timesteps += 1
        reward = self.compute_reward()
        done = True if self.timesteps >= self.time_horizon else False
        return self._get_obs(), reward, 0, done, {}

    def compute_reward(self):
        agent_pos_tuple = tuple(self.agent_pos)

        reward = np.array([0, 0])  # Initialize reward as [0, 0]

        # Check if agent is on a resource
        if agent_pos_tuple in self.resource_positions:
            index = self.resource_positions.index(agent_pos_tuple)
            if self.resources_status[index] == 1:
                self.collected_resources += 1
                reward[0] = 1  # Collecting a resource
                self.resources_status[index] = 0

        # Check if agent is on an enemy
        if agent_pos_tuple in self.enemy_positions:
            self.damage_taken += 1
            reward[1] = 1  # Encountering an enemy

        return reward

    def render(self, mode='human', save_path='./env_render.png'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])
        ax.set_xticks(np.arange(0, self.grid_size[1] + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size[0] + 1, 1))
        ax.grid(True)

        # Draw grid
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                ax.add_patch(patches.Rectangle((x, self.grid_size[0] - y - 1), 1, 1, edgecolor='black', facecolor='none'))

        # Draw resources
        for idx, pos in enumerate(self.resource_positions):
            if self.resources_status[idx] == 1:
                ax.add_patch(patches.Rectangle((pos[1], self.grid_size[0] - pos[0] - 1), 1, 1, color='green'))

        # Draw enemies
        for pos in self.enemy_positions:
            ax.add_patch(patches.Rectangle((pos[1], self.grid_size[0] - pos[0] - 1), 1, 1, color='red'))

        # Draw agent last to ensure it's on top
        ax.add_patch(patches.Rectangle((self.agent_pos[1], self.grid_size[0] - self.agent_pos[0] - 1), 1, 1, edgecolor='black', facecolor='blue'))

        # Set labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, self.grid_size[1] + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, self.grid_size[0] + 1, 1), minor=True)
        ax.grid(which='both')

        plt.savefig(save_path, format='png')
        plt.close(fig)

    def decode_state(self, state):
        # Extract resource status
        resource_status_int = state % (2 ** self.num_resources)
        # Extract agent position
        agent_position_flat = state // (2 ** self.num_resources)

        # Convert agent position to 2D coordinates
        x = agent_position_flat // self.grid_size[1]
        y = agent_position_flat % self.grid_size[1]
        agent_pos = [x, y]

        # Convert resource status integer to binary vector
        resource_status_bin = format(resource_status_int, f'0{self.num_resources}b')
        resource_status = np.array(list(map(int, resource_status_bin)), dtype=np.int32)

        return agent_pos, resource_status
    
    def get_transition(self, state: int):
        """
        Given a state (s), return all the transition probabilities Pr(s'|s,a) and R(s,a) for all possible actions.

        Parameters
        ----------
        state : int
            integer encoding of the state
            
        Returns
        -------
        transition_prob : array
            transition probability for each action
        reward_arr : array
            reward for each action
        next_state_arr : array
            next state for each action
        """
        # Decode the state
        agent_pos, resource_status = self.decode_state(state)
        transition_prob = np.zeros(self.action_space.n)
        reward_arr = np.zeros((self.action_space.n, 2))  # Reward is now a vector [resources_collected, damage_taken]
        next_state_arr = np.zeros(self.action_space.n, dtype=int)

        # For each action, calculate transition probability, reward, and next state
        for action in range(self.action_space.n):
            # Copy current state
            new_agent_pos = agent_pos.copy()
            new_resource_status = resource_status.copy()

            # Determine new position based on action
            if action == 0 and new_agent_pos[0] > 0:  # up
                new_agent_pos[0] -= 1
            elif action == 1 and new_agent_pos[0] < self.grid_size[0] - 1:  # down
                new_agent_pos[0] += 1
            elif action == 2 and new_agent_pos[1] > 0:  # left
                new_agent_pos[1] -= 1
            elif action == 3 and new_agent_pos[1] < self.grid_size[1] - 1:  # right
                new_agent_pos[1] += 1

            # Calculate reward
            reward = np.array([0, 0])  # Initialize reward as [0, 0]
            new_pos_tuple = tuple(new_agent_pos)
            if new_pos_tuple in self.resource_positions:
                index = self.resource_positions.index(new_pos_tuple)
                if new_resource_status[index] == 1:
                    reward[0] = 1  # Collecting a resource
                    new_resource_status[index] = 0  # Resource is now collected
            if new_pos_tuple in self.enemy_positions:
                reward[1] = 1  # Encountering an enemy

            # Encode new state
            new_agent_flat = new_agent_pos[0] * self.grid_size[1] + new_agent_pos[1]
            new_resource_status_int = int(''.join(map(str, new_resource_status)), 2)
            next_state = new_agent_flat * (2 ** self.num_resources) + new_resource_status_int

            # Update arrays
            transition_prob[action] = 1  # Deterministic transitions
            reward_arr[action] = reward
            next_state_arr[action] = next_state

        return transition_prob, reward_arr, next_state_arr