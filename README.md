# Multi-objective Reinforcement Learning with Nonlinear Preferences: Provable Approximation for Maximizing Expected Scalarized Return
[![arxiv badge](https://img.shields.io/badge/arXiv-2311.02544-red)](http://www.arxiv.org/abs/2311.02544)
This repository contains the code implementation of our proposed algorithm, *Reward-Aware Value Iteration*, and the related experiments discussed in our paper. Our work is accepted by [AAMAS 2025](https://aamas2025.org/).

## Dependencies ✅
* See `requirements.txt` for `pip`
* Alternatively, see `env.yml` for `conda`

## Project Structure 📚
* `algo` contains the implementation of all the algorithms discussed in our paper, specifically:
    * `algo/linear_scalarize.py`: linearly scalarized policy
    * `algo/mixture.py`: mixture policy
    * `algo/ra_value_iteration_cpu.py`: CPU implementation of Reward-Aware Value Iteration (RAVI) algorithm
    * `algo/ra_value_iteration_gpu.py`: GPU implementation of RAVI
    * `algo/utils.py`: helper functions and classes, also contains implementation of welfare functions
    * `algo/vanilla_value_iteration.py`: single-objective value iteration algorithm, used as a subroutine for model-based mixture policy baseline discussed in the paper.
    * `algo/welfare_q.py`: Implementation of Welfare Q-Learning
* `constants` contains the hyperparameters of all the algorithms
* `envs` contains the simulation environments for Taxi (`envs/fair_taxi.py`) and Scavenger (`envs/Resource_Gathering.py`)
* `results` is an empty folder for storing results
* `results.py` is a script for analyzing the results
* `train.py` is the main script responsible to initialize the algorithm runs. We have left descriptions for each command-line arguments in the script

## More About RAVI 🛠
As discussed in our [paper](http://www.arxiv.org/pdf/2311.02544), RAVI is parallelizable. Therefore, for both CPU and GPU implementations, we have parallel and serial versions of RAVI. For the CPU version, we have both multi-threading and multi-processing available. We have found that multi-threading implementation works better.


## Citation
```bibtex
@article{peng2023nonlinear,
  title={Nonlinear Multi-objective Reinforcement Learning with Provable Guarantees},
  author={Peng, Nianli and Fain, Brandon},
  journal={arXiv preprint arXiv:2311.02544},
  year={2023}
}
```
