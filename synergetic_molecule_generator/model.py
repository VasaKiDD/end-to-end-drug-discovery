import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase

import synergetic_molecule_generator.crossover as co
import synergetic_molecule_generator.mutate as mu

rdBase.DisableLog("rdApp.error")

# parent_A = Chem.MolFromSmiles('O=C(CNC(=O)C1=NN2C=CC=CC2=C1)NCCN1CCOCC1')
# parent_B = Chem.MolFromSmiles('CCNC1=NN=C(C2=CC=CC(S(C)(=O)=O)=C2)S1')

import deep_likeliness.tokenizer as tokenizer


def reproduce(args):

    mating_pool, population_size, mutation_rate = args
    new_population = []
    while len(new_population) < population_size:
        parent_A = Chem.MolFromSmiles(random.choice(mating_pool))
        parent_B = Chem.MolFromSmiles(random.choice(mating_pool))
        new_child = co.crossover(parent_A, parent_B)
        if new_child != None:
            mutated_child = mu.mutate(new_child, mutation_rate)
            if mutated_child != None:
                try:
                    s = Chem.MolToSmiles(mutated_child)
                    tokens = [1] + [tokenizer.__t2i[tok] for tok in tokenizer.smiles_tokenizer(s)]
                    new_population.append(s)
                except Exception as e:
                    print(e)

    return new_population


class GAMoleculeModel:
    def __init__(self, mutation_rate, n_cpu):
        self.mutation_rate = mutation_rate

        self.dataset = pd.read_csv("data/dataset_v1.csv")
        self.dataset = self.dataset.to_numpy()
        self.dataset = self.dataset[:, 0]
        self.n_cpu = n_cpu

    def predict_batch(self, states, rewards):  # seems to be the best

        rewards = rewards / np.sum(rewards)

        for i in range(len(states)):
            if states[i] is None:
                states[i] = np.random.choice(self.dataset)

        # mutating_size = int(len(states) * 0.1)
        # idx = (-rewards).argsort()[:mutating_size]
        # mutating_population = np.array(states)[idx].tolist()
        # population_size = len(states) - mutating_size
        # reinit_child = [np.random.choice(self.dataset)] if population_size > 1 else []
        # new_childs = reproduce(mutating_population, population_size, self.mutation_rate)
        # states = mutating_population + new_childs #+ reinit_child
        # return states

        # 16s

        mutating_population = np.random.choice(states, len(states), p=rewards).tolist()

        args = []
        n_walkers = int(len(states) // self.n_cpu)
        for i in range(self.n_cpu):
            args.append(
                [
                    mutating_population[i * n_walkers : (i + 1) * n_walkers],
                    n_walkers,
                    self.mutation_rate,
                ]
            )

        with Pool(self.n_cpu) as pool:
            output = pool.map(reproduce, args)

        new_childs = []
        for states in output:
            new_childs.extend(states)

        return new_childs
