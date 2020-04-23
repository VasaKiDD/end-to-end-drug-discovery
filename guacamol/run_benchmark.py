import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guacamol.synergetic_goal_directed_generator import SynergeticMolGDG
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from deep_likeliness.discriminator import RNNEncoder

import torch

if __name__ == "__main__":

    n_walkers = 2048
    n_cpu = 8
    mutation_rate = 0.05  # 0.05
    iterations = 100
    balance = 1.0
    version = "v2"
    accumulate_rewards = False

    json_file = "guacamol/json_results/first_nn_trial.json"

    # neural net
    hidden_size = 256
    num_layers = 2
    biderectional = False
    file = "deep_likeliness/saved_parameters/deep_scorer_network.pth"
    neural_net = RNNEncoder(
        hidden_size=hidden_size, num_layers=num_layers, bidirectional=biderectional
    )
    neural_net.load_state_dict(torch.load(file))

    model = SynergeticMolGDG(
        neural_network=neural_net,
        mutation_rate=mutation_rate,
        n_walkers=n_walkers,
        balance=balance,
        iterations=iterations,
        accumulate_rewards=accumulate_rewards,
        n_cpu=n_cpu,
    )

    assess_goal_directed_generation(model, json_output_file=json_file, benchmark_version=version)
