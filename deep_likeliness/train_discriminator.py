import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from synergetic_molecule_generator.model import GAMoleculeModel
from synergetic_molecule_generator.swarm import Swarm
from deep_likeliness.dataloader import MolecularDataset
from deep_likeliness.discriminator import RNNEncoder
import torch.optim as optim
import os
import torch
import numpy as np
import pandas as pd
from synergetic_molecule_generator.molecules import penalized_logP


class logpscorer:
    def __init__(self):
        pass

    def score_list(self, states):
        scores = []
        for s in states:
            scores.append(penalized_logP(s))
        return scores


if __name__ == "__main__":

    # Env parameters
    mutation_rate = 0.05

    # Swarm parameters
    n_walkers = 1024
    number_molecules = 50000
    balance = 1
    n_cpu = 8

    # Dataloader
    batch_size = 64
    number_swarms = 40

    # NeuralNetwork
    hidden_size = 256
    num_layers = 2
    biderectional = False

    # optimiser
    learning_rate = 0.01

    # saving paramters
    parameter_file = "deep_scorer"

    # Dataset to verify existing molecules
    dataset = pd.read_csv("data/dataset_v1.csv")
    dataset = dataset.to_numpy()
    dataset = dataset[:, 0]
    smile_dic = {}
    for smile in dataset:
        smile_dic[smile] = True
    dataset = smile_dic

    model = GAMoleculeModel(mutation_rate=mutation_rate, n_cpu=n_cpu)

    md = MolecularDataset(
        sources=[
            {
                "path": "data/train_plogp_plogpm.csv",
                "smiles": "SMILES",
                "prob": 1,
                "plogP": "plogP",
            }
        ],
        props=["plogP"],
    )

    true_loader = DataLoader(
        md, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    deep_scorer = RNNEncoder(
        hidden_size=hidden_size, num_layers=num_layers, bidirectional=biderectional
    )

    if os.path.exists("deep_likeliness/saved_parameters/%s_network.pth" % parameter_file):
        print("Model found ! Loading model...")
        deep_scorer.load_state_dict(
            torch.load("deep_likeliness/saved_parameters/%s_network.pth" % parameter_file)
        )
    else:
        print("New model")

    discriminator_optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, deep_scorer.parameters()), lr=learning_rate
    )

    binary_loss = torch.nn.BCEWithLogitsLoss()

    scorer = logpscorer()

    wave = Swarm(
        model=model,
        neural_network=deep_scorer,
        scoring_function=None,
        n_walkers=n_walkers,
        render_every=1,
        balance=balance,
        accumulate_rewards=False,
    )

    for s_i in range(number_swarms):
        print("Number :", s_i + 1)

        states, rewards = wave.run_swarm_training(number_molecules, reset=True, print_swarm=True)
        states = states.tolist()

        for sm in states:
            if sm in dataset.keys():
                states.remove(sm)

        print("data generated :", len(states))
        if len(rewards.shape) == 1:
            print("Rewards mean", rewards.mean())
        else:
            print("Rewards nn mean:", rewards[:, 1].mean())
            print("Rewards 2 mean:", rewards[:, 0].mean())

        batch = len(states) // batch_size
        i = 0
        perm = np.random.permutation(len(states))
        states = np.array(states)[perm].tolist()
        for x_true, _ in true_loader:
            x_true = list(x_true)
            true_labels = torch.ones(batch_size, 1)
            false_labels = torch.zeros(batch_size, 1)
            labels = torch.cat((true_labels, false_labels), dim=0)
            x_true.extend(states[i * batch_size : (i + 1) * batch_size])
            preds, _ = deep_scorer.encode(x_true)

            with torch.no_grad():
                accuracy1 = torch.where(
                    preds[:batch_size] > 0,
                    torch.ones_like(preds[:batch_size]),
                    torch.zeros_like(preds[:batch_size]),
                )
                accuracy2 = torch.where(
                    preds[batch_size:] < 0,
                    torch.ones_like(preds[batch_size:]),
                    torch.zeros_like(preds[batch_size:]),
                )
                accuracy1 = accuracy1.sum() / batch_size
                accuracy2 = accuracy2.sum() / batch_size

            print("True positives :", accuracy1)
            print("False neguative :", accuracy2)

            loss = binary_loss(preds, labels)
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()

            print("Loss :", float(loss))

            i += 1
            if i == batch:
                break

        torch.save(
            deep_scorer.state_dict(),
            "deep_likeliness/saved_parameters/%s_network.pth" % parameter_file,
        )
