from typing import List, Optional

import numpy as np

# from synergetic_molecule_generator.swarm_with_cloning import Swarm
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from synergetic_molecule_generator.model import GAMoleculeModel
from synergetic_molecule_generator.swarm import Swarm


class SynergeticMolGDG(GoalDirectedGenerator):
    def __init__(
        self,
        neural_network=None,
        mutation_rate=0.0,
        n_walkers=1024,
        balance=1,
        iterations=100,
        accumulate_rewards=False,
        n_cpu=8,
    ):

        self.env = None

        self.model = GAMoleculeModel(mutation_rate, n_cpu=n_cpu)

        # self.model = MoleculeModelTokens()

        self.iterations = iterations
        self.n_walker = n_walkers
        self.balance = balance
        self.accumulate_rewards = accumulate_rewards

        self.neural_network = neural_network if neural_network else None

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Given an objective function, generate molecules that score as high as possible.

        Args:
            scoring_function: scoring function
            number_molecules: number of molecules to generate
            starting_population: molecules to start the optimization from (optional)

        Returns:
            A list of SMILES strings for the generated molecules.
        """

        if starting_population is not None:
            state = starting_population[0]
        else:
            state = None

        wave = Swarm(
            model=self.model,
            neural_network=self.neural_network,
            scoring_function=scoring_function,
            n_walkers=self.n_walker,  # self.n_walker
            render_every=1,
            balance=self.balance,
            accumulate_rewards=self.accumulate_rewards,
        )  # False

        states, rewards = wave.run_swarm(iterations=self.iterations, state=state, print_swarm=True)

        if len(rewards.shape) > 1:
            idx = (-rewards[:, 0]).argsort()[:number_molecules]
            max_states = np.array(states)[idx]
            max_rewards = rewards[idx]
        else:
            idx = (-rewards).argsort()[:number_molecules]
            max_states = np.array(states)[idx]
            max_rewards = rewards[idx]

        return max_states.tolist()
