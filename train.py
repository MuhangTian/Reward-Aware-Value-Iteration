import argparse
import logging
import gymnasium as gym
import numpy as np
import wandb

from algo.linear_scalarize import LinearScalarize, LinearScalarizeM
from algo.mixture import MixturePolicy, MixturePolicyM
from constants.hyperparam import scaling_factors, linscal_weights, mixture_weights
from algo.utils import (is_file_not_on_disk, is_positive_float,
                        is_positive_integer, is_within_zero_one_float,
                        add_time_suffix, is_file_on_disk, seed_everything)
from algo.welfare_q import WelfareQ
from envs.fair_taxi import Fair_Taxi_MOMDP
from envs.Resource_Gathering import ResourceGatheringEnv


logging.basicConfig(format=("[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

YOUR_WANDB_NAME = ""        # OPTIONAL: your wandb entity name, wandb: https://wandb.ai/site/ 
YOUR_WANDB_KEY = ""         # OPTIONAL: your wandb key


def get_setting(size, num_locs):
    """
    To store environment settings

    Parameters
    ----------
    size : int
        size of the grid world in N x N
    num_locs : int
        number of location destination pairs
    """
    if num_locs == 2:
        loc_coords = [[0,0],[3,2]]
        dest_coords = [[0,3],[3,3]]

    elif num_locs == 1:
        loc_coords = [[0,0]]
        dest_coords = [[1,1]]

    elif num_locs == 3:
        loc_coords = [[0,0],[3,2],[1,0]]
        dest_coords = [[0,3],[3,3],[0,1]]

    elif num_locs == 4:
        loc_coords = [[4, 7], [6, 6], [8, 3], [8, 9]]
        dest_coords = [[2, 7], [4, 5], [1, 8], [9, 2]]

    elif num_locs == 5:
        loc_coords = [[0, 0], [3, 2], [1, 0], [4, 4], [2, 3]]
        dest_coords = [[0, 3], [3, 3], [0, 1], [4, 1], [9, 9]]
        
    else:
        raise NotImplementedError("Number of locations not implemented")
    
    return size, loc_coords, dest_coords

def parse_arguments():
    prs = argparse.ArgumentParser()
    prs.add_argument(
        '--env_name', 
        choices=["Fair_Taxi_MOMDP", "ResourceGatheringEnv"], 
        default="Fair_Taxi_MOMDP",
        help="Environment to train on",
    )
    prs.add_argument(
        '--size', 
        type=is_positive_integer, 
        default=10,
        help="Size of the grid world",
    )
    prs.add_argument(
        '--num_locs', 
        type=is_positive_integer, 
        default=3,
        help="Number of location destination pairs",
    )
    prs.add_argument(
        '--time_horizon', 
        type=is_positive_integer, 
        default=30,
        help="Time horizon for the environment",
    )
    prs.add_argument(
        '--discre_alpha', 
        type=is_positive_float, 
        default=1,
        help="discretization factor alpha for RAVI",
    )
    prs.add_argument(
        "--gamma", 
        type=is_positive_float, 
        default=1,
        help="Discount factor for the environment",
    )
    prs.add_argument(
        "--growth_rate", 
        type=is_positive_float, 
        default=1.0,
        help="Growth rate for RAVI",
    )
    prs.add_argument(
        "--welfare_func_name", 
        choices=["egalitarian", "nash-welfare", "p-welfare", "RD-threshold", "Cobb-Douglas", "utilitarian"], 
        default="nash-welfare",
        help="Welfare function to optimize",
    )
    prs.add_argument(
        "--nsw_lambda",
        type=is_positive_float, 
        default=1e-4,
        help="NSW lambda for Nash social welfare",
    )
    prs.add_argument(
        "--wandb", 
        action="store_true",
        help="whether log to wandb",
    )
    prs.add_argument(
        "--save_path", 
        type=is_file_not_on_disk, 
        default="results/trial.npz",
        help="Path to save the results",
    )
    prs.add_argument(
        "--method", 
        choices=["welfare_q", "ravi-cpu", "ravi-cpu-prl", "ravi-gpu", "linear_scalarize", "mixture", "linear_scalarize_m", "mixture_m"], 
        default="ravi-cpu-prl",
        help="Method to train the agent",
    )
    prs.add_argument(
        "--lr", 
        type=is_positive_float, 
        default=1e-3,
        help="Learning rate for the agent",
    )
    prs.add_argument(
        "--epsilon", 
        type=is_within_zero_one_float, 
        default=0.1,
        help="Epsilon for epsilon-greedy policy",
    )
    prs.add_argument(
        "--episodes", 
        type=is_positive_integer, 
        default=1000,
        help="Number of episodes to train the agent (this only applies to online algorithms)",
    )
    prs.add_argument(
        "--init_val", 
        type=float, 
        default=0.0,
        help="Initial value for the Q-table",
    )
    prs.add_argument(
        "--dim_factor", 
        type=is_positive_float, 
        default=0.9,
        help="diminishing factor for epsilon-greedy policy for Welfare Q-Learning",
    )
    prs.add_argument(
        "--p", 
        type=float, 
        default=0.5,
        help="p parameter for p-welfare function",
    )
    prs.add_argument(
        "--threshold", 
        type=is_positive_float, 
        default=4, 
        help="for resource damage scalarization",
    )
    prs.add_argument(
        "--scaling_factor", 
        type=int, 
        default=1,
        help="Scaling factor for the reward initialization for RAVI, used to save memory consumption",
    )
    prs.add_argument(
        "--auto_scaling", 
        action="store_true", 
        help="whether to automatically scale the reward initialization for RAVI based on tuned values",
    )
    prs.add_argument(
        "--num_resources", 
        type=int, 
        default=6, 
        help="number of resources in Scavenger",
    )
    prs.add_argument(
        "--seed", 
        type=int, 
        default=1122,
        help="Seed for reproducibility",
    )
    prs.add_argument(
        "--wdb_entity", 
        type=str, 
        default="",
        help="OPTIONAL: wandb entity name",
    )
    prs.add_argument(
        "--project", 
        type=str, 
        default="RAVI-Experiments",
        help="OPTIONAL: wandb project name",
    )
    prs.add_argument(
        "--eval", 
        action="store_true",
        help="whether to run RAVI in evaluation mode",
    )
    prs.add_argument(
        "--rho", 
        default=0.4, 
        type=float, 
        help="rho parameter for Cobb-Douglas welfare function",
    )
    prs.add_argument(
        "--load_path", 
        type=is_file_on_disk, 
        default=None,
        help="Path to load the trained model",
    )
    return prs.parse_args()

def key_based_on_name(name):
    if name == YOUR_WANDB_NAME:
        return YOUR_WANDB_KEY
    else:
        raise ValueError(f"Invalid entity name: {name}")

def init_if_wandb(args):
    if args.wandb:
        wandb.login(key=key_based_on_name(args.wdb_entity))
        wandb.init(project=args.project, entity=args.wdb_entity)
        wandb.config.update(args)

def scaling_factor(args):
    if args.auto_scaling:
        if args.welfare_func_name == "p-welfare":
            return scaling_factors[f"{args.welfare_func_name}-{args.p}_{args.num_locs}"]
        else:
            return scaling_factors[f"{args.welfare_func_name}_{args.num_locs}"]
    return args.scaling_factor

if __name__ == "__main__":
    args = parse_arguments()
    if args.env_name == "Fair_Taxi_MOMDP":
        size, loc_coords, dest_coords = get_setting(args.size, args.num_locs)
        env = Fair_Taxi_MOMDP(
            size = size, 
            loc_coords = loc_coords, 
            dest_coords = dest_coords, 
            fuel = args.time_horizon, 
            output_path = '', 
            fps = 15
        )
    elif args.env_name == "ResourceGatheringEnv":
        env = ResourceGatheringEnv(
            time_horizon = args.time_horizon, 
            grid_size = (args.size, args.size), 
            num_resources = args.num_resources, 
            seed = args.seed
        )
    
    seed_everything(args.seed)
    args.scaling_factor = scaling_factor(args)
    init_if_wandb(args)
    
    if args.method == "ravi-gpu":
        from algo.ra_value_iteration_gpu import RAValueIterationGPU 
        algo = RAValueIterationGPU(
            env = env,
            discre_alpha = args.discre_alpha,
            gamma = args.gamma,
            growth_rate = args.growth_rate,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            p = args.p,
            threshold = args.threshold,
            scaling_factor = args.scaling_factor,
            rho = args.rho,
        )
    
    elif args.method == "ravi-cpu":
        from algo.ra_value_iteration_cpu import RAValueIterationCPU
        algo = RAValueIterationCPU(
            env = env,
            discre_alpha = args.discre_alpha,
            gamma = args.gamma,
            growth_rate = args.growth_rate,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            p = args.p,
            threshold = args.threshold,
            scaling_factor = args.scaling_factor,
            parallel = False,
            eval = args.eval,
            load_path = args.load_path,
            seed = args.seed,
            rho = args.rho,
        )

    elif args.method == "ravi-cpu-prl":
        from algo.ra_value_iteration_cpu import RAValueIterationCPU
        algo = RAValueIterationCPU(
            env = env,
            discre_alpha = args.discre_alpha,
            gamma = args.gamma,
            growth_rate = args.growth_rate,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            p = args.p,
            threshold = args.threshold,
            scaling_factor = args.scaling_factor,
            parallel = True,
            eval = args.eval,
            load_path = args.load_path,
            seed = args.seed,
            rho = args.rho,
        )
        
    elif args.method == "welfare_q":
        algo = WelfareQ(
            env = env,
            lr = args.lr,
            gamma = args.gamma,
            epsilon = args.epsilon,
            episodes = args.episodes,
            init_val = args.init_val,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            dim_factor = args.dim_factor,
            p = args.p,
            threshold = args.threshold,
            rho = args.rho,
        )

    elif args.method == "linear_scalarize":
        algo = LinearScalarize(
            env = env,
            init_val = args.init_val,
            episodes = args.episodes,
            weights = linscal_weights[str(args.num_locs)],
            lr = args.lr,
            gamma = args.gamma,
            epsilon = args.epsilon,
            welfare_func_name = args.welfare_func_name,
            save_path = args.save_path,
            nsw_lambda = args.nsw_lambda,
            p = args.p,
            wdb = args.wandb,
            threshold = args.threshold,
            rho = args.rho,
        )

    elif args.method == "mixture":
        algo = MixturePolicy(
            env = env,
            episodes = args.episodes,
            time_horizon = args.time_horizon,
            lr = args.lr,
            epsilon = args.epsilon,
            gamma = args.gamma,
            init_val = args.init_val,
            weights = mixture_weights[str(args.num_locs)],
            interval = 1,   # change policy after t/d timesteps
            welfare_func_name = args.welfare_func_name,
            save_path = args.save_path,
            nsw_lambda = args.nsw_lambda,
            p = args.p,
            wdb = args.wandb,
            threshold = args.threshold,
            rho = args.rho,
        )
    
    elif args.method == "linear_scalarize_m":
        algo = LinearScalarizeM(
            env = env,
            reward_dim = args.num_locs,
            gamma = args.gamma,
            epsilon = args.epsilon,
            save_path = args.save_path,
            welfare_func_name = args.welfare_func_name,
            p = args.p,
            episodes = args.episodes,
            nsw_lambda = args.nsw_lambda,
            lr = args.lr,
            weights = linscal_weights[str(args.num_locs)],
            seed = args.seed,
            tol = 1e-6,
            log_every = 100,
            wdb = args.wandb,
            val_load_path = args.load_path,
            threshold = args.threshold,
            rho = args.rho,
        )
    
    elif args.method == "mixture_m":
        algo = MixturePolicyM(
            env = env,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            gamma = args.gamma,
            interval = 1,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            val_load_path = args.load_path,
            p = args.p,
            seed = args.seed,
            wdb = args.wandb,
            tol = 1e-6,
            log_every = 100,
            threshold = args.threshold,
            rho = args.rho,
        )

    
    algo.train()