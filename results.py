"""Produce results for the paper."""
import numpy as np
import argparse
from tqdm.rich import tqdm
import seaborn as sns

def prs_args():
    prs = argparse.ArgumentParser()
    prs.add_argument("--method", type=str, required=True, help="Name of the method")
    prs.add_argument("--algo", type=str, default="ravi", help="Name of the algorithm")
    prs.add_argument("--d", type=int, default=2, help="dimension of the reward function")
    prs.add_argument("--n", type=int, default=15, help="size of the grid world")
    prs.add_argument("--welf_func", type=str, default="nash-welfare", help="name of the welfare function")
    prs.add_argument("--p", type=str, default=10, help="p parameter for p-welfare")
    return prs.parse_args()

def print_online_algs(algo: str, d: int, n: int, welf_func: str, max_iter: int=10):
    result = []
    for i in range(1, max_iter+1):
        if welf_func in ["RD-threshold", "Cobb-Douglas"]:
            data = np.load(f"results/{algo}/{welf_func}/{i}.npz")
        else:
            data = np.load(f"results/{algo}/{welf_func}/{n}-{d}-{i}.npz")
        score = data["nonlinear_record"]
        print(score)
        result.append(score[-1])
    print(f"Algo = {algo} | Welfare func = {welf_func} | d = {d} | n = {n} | {np.mean(result):.3f}$\pm${np.std(result):.3f}")

def plot_nonlinear_over_time_graph(algos: list[str], d: int, n: int, welf_func: str, p, max_iter: int=10, alpha=0.8, linewidth=0.8, episodes=500_000):
    import matplotlib.pyplot as plt
    import os
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    os.makedirs("results/figs", exist_ok=True)

    for algo in algos:
        y, yerr = [], []
        pbar = tqdm(range(1, max_iter+1))
        for i in pbar:
            pbar.set_description(f"Calculating for {algo}...")
            try:
                if welf_func != "p-welfare":
                    data = np.load(f"results/{algo}/{welf_func}/{n}-{d}-{i}.npz")
                else:
                    data = np.load(f"results/{algo}/{welf_func}-{p}/{n}-{d}-{i}.npz")
                score = data["nonlinear_record"]
                y.append(score)
            except FileNotFoundError:
                continue

        y = np.array(y)[:,:episodes]
        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        x = np.arange(len(y))
        plt.plot(x, y, label=algo, alpha=alpha, linewidth=linewidth)
        plt.fill_between(x, y - yerr, y + yerr, alpha=alpha-0.05)  # Adjust alpha for transparency

    plt.ylim(bottom=0)
    plt.xlabel("Episodes")
    plt.ylabel("Welfare")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/figs/{welf_func}-{n}-{d}.png")
    print(f"Complete, figure is saved at \"results/figs/{welf_func}-{n}-{d}.png\"")

if __name__ == "__main__":
    args = prs_args()

    if args.method == "mean_std":
        print_online_algs(args.algo, args.d, "15", args.welf_func)
    
    elif args.method == "plot":
        plot_nonlinear_over_time_graph(
            algos = ["mixture", "welfare_q", "linscal"],
            d = args.d,
            n = args.n,
            welf_func = args.welf_func,
            p = args.p,
        )
    
    else:
        raise ValueError(f"{args.method} method does not exist!")