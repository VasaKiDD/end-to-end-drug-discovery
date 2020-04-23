import copy
from typing import Iterable

import numpy as np
from IPython.core.display import clear_output
from SynergeticGenerator import molecules
from rdkit import DataStructs
from tqdm import tqdm


def normalize_vector(vector: np.ndarray):
    avg = vector.mean()
    if avg == 0:
        return np.ones(len(vector))
    standard = vector / avg
    return standard


def _relativize_vector(vector: np.ndarray):
    std = vector.std()
    if std == 0:
        return np.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = np.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = np.exp(standard[standard <= 0])
    return standard


def relativize_vector(vector: np.ndarray):
    std = vector.std()
    if std == 0:
        return np.ones(len(vector))
    standard = (vector - vector.mean()) / std
    return np.exp(standard)


def normalize_vector_zero_one(vector):
    """
    Returns normalized values where min = 0 and max = 1.
    :param vector: array to be normalized.
    :return: Normalized vector.
    """
    max_r, min_r = np.max(vector), np.min(vector)
    if min_r == max_r:
        _reward = np.ones(len(vector), dtype=np.float32)
    else:
        _reward = (vector - min_r) / (max_r - min_r)
    return _reward


class DataStorage:
    """This is a class for storing the states and the observations of a Swarm.
    This way is slower than storing it in a numpy array, but it allows to store
    any kind of states and observations."""

    def __init__(self):
        self.states = {}
        self.actions = {}
        self.walker_ids = []

    def __getitem__(self, item):
        states = self.get_states(item)
        actions = self.get_actions(item)
        return states, actions

    def reset(self):
        self.states = {}
        self.actions = {}
        self.walker_ids = []

    def get_states(self, labels: Iterable) -> list:
        return [self.states[label] for label in labels]

    def get_actions(self, labels: Iterable) -> list:
        return [self.actions[label] for label in labels]

    def append(self, walker_ids: [list, np.ndarray], states: Iterable, actions):
        actions = actions if actions is not None else [None] * len(walker_ids)
        for w_id, state, action in zip(walker_ids, states, actions):
            if w_id not in self.walker_ids:
                self.states[w_id] = copy.deepcopy(state)
                if actions is not None:
                    self.actions[w_id] = copy.deepcopy(action)
        self.walker_ids = list(set(self.walker_ids))
        self.walker_ids += list(set(walker_ids))

    def update_values(self, walker_ids):
        # This is not optimal, but ensures no memory leak
        new_states = {}
        new_actions = {}
        walker_ids = list(walker_ids)
        walker_ids.append(0)
        for w_id in walker_ids:
            new_states[w_id] = self.states[w_id]
            new_actions[w_id] = self.actions[w_id]
        self.states = new_states
        self.actions = new_actions
        self.walker_ids = walker_ids


class Swarm:

    """This is the most basic mathematical entity that can be derived from Fractal AI theory.
    It represents a cloud of points that propagates through an state space. Each walker of the
    swarm evolves by either cloning to another walker or perturbing the environment.
    """

    def __init__(
        self,
        model,
        scoring_function=None,
        neural_network=None,
        n_walkers: int = 100,
        balance: float = 1.0,
        render_every: int = 1e10,
        accumulate_rewards: bool = True,
    ):
        """

        :param env: Environment that will be sampled.
        :param model: Model used for sampling actions from observations.
        :param n_walkers: Number of walkers that the swarm will use
        :param balance: Balance coefficient for the virtual reward formula.
        :param render_every: Number of iterations that will be performed before printing the Swarm
         status.
        :param accumulate_rewards: Use the accumulated reward when scoring the walkers.
                                  False to use instantaneous reward.
        """

        # Parameters of the algorithm
        self._model = model
        self.n_walkers = n_walkers
        self.balance = balance
        self.render_every = render_every
        self.accumulate_rewards = accumulate_rewards

        # Environment information sources
        self.observations = None
        self.rewards = np.zeros(self.n_walkers)
        # Internal masks
        # True when the boundary condition is met
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        # Walkers that will clone to a random companion
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        self.not_frozen = np.ones(self.n_walkers, dtype=bool)
        # If true the corresponding walker will not move
        # Processed information sources
        self._clone_idx = None
        self.times = np.zeros(self.n_walkers)
        self.dt = np.ones(self.n_walkers, dtype=int)
        self._n_samples_done = 0
        self._i_simulation = 0
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        self._virtual_reward = np.ones(self.n_walkers)
        # This is for storing states and actions of arbitrary shape and type
        self.data = DataStorage()
        self._pre_clone_ids = [0]
        self._post_clone_ids = [0]
        self.all_data = {}
        self.scoring_function = scoring_function if scoring_function is not None else None
        self.neural_network = neural_network if neural_network is not None else None

    def __str__(self):
        """Print information about the internal state of the swarm."""

        rewards = []
        for key in self.all_data.keys():
            rewards.append(self.all_data[key])
        rewards = np.array(rewards)

        if self.neural_network is not None and self.scoring_function is not None:
            idx = (-rewards[:, 0]).argsort()[: self.n_walkers]
            self.print_rewards = rewards[idx]
            self.print_rewards = self.print_rewards[:, 0]
        else:
            idx = (-rewards).argsort()[: self.n_walkers]
            self.print_rewards = rewards[idx]

        text = (
            "Environment: {} | Walkers: {} | clones {}\n"
            "Total samples: {}\n"
            "Reward: mean {:.2f} | Dispersion: {:.2f} | max {:.2f} | min {:.2f} | std {:.2f}\n"
            "Episode length: {:.2f}\n"
            "Number unique molecules : {}".format(
                "GA_molecule",
                self.n_walkers,
                self._will_clone.sum(),
                self._n_samples_done,
                self.print_rewards.mean(),
                self.print_rewards.max() - self.print_rewards.min(),
                self.print_rewards.max(),
                self.print_rewards.min(),
                self.print_rewards.std(),
                self.times.mean(),
                len(self.all_data),
            )
        )
        return text

    @property
    def actions(self):
        return self.data.get_actions(self.walkers_id)

    @property
    def states(self):
        return self.data.get_states(self.walkers_id)

    def init_swarm(self, state: np.ndarray = None, obs: np.ndarray = None):
        """
        Synchronize all the walkers to a given state, and clear all the internal data of the swarm.
        :param state: State that all the walkers will copy. If None, a new game is started.
        :param obs: Observation corresponding to state. If None, a new game is started.
        :return:
        """
        if state is not None:
            self.observations = [molecules.get_fingerprint(state) for _ in range(self.n_walkers)]
        else:
            self.observations = np.array([None for _ in range(self.n_walkers)])

        # Environment Information sources

        self.rewards = np.zeros(self.n_walkers, dtype=np.float32)
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        # Internal masks
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        self.not_frozen = np.ones(self.n_walkers, dtype=bool)
        # Processed information sources
        self._virtual_reward = np.ones(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self.dt = np.ones(self.n_walkers, dtype=int)
        self._n_samples_done = 0
        self._i_simulation = 0
        self.data = DataStorage()
        self.data.reset()
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        # Store data and keep indices

        states = np.array([copy.deepcopy(state) for _ in range(self.n_walkers)])
        # actions = self._model.predict_batch(states, self.not_frozen)
        self.data.append(
            walker_ids=self.walkers_id,
            states=states,
            actions=[None for _ in range(self.n_walkers)],
        )
        self._clone_idx = None
        self._pre_clone_ids = [0]
        self._post_clone_ids = [0]
        self.all_data = {}

    def step_walkers(self):
        """Sample an action for each walker, and act on the environment. This is how the Swarm
        evolves.
        :return: None.
        """

        states = self.data.get_states(self.walkers_id)

        new_states = self._model.predict_batch(states, self._virtual_reward)

        if self.neural_network is not None:
            drug_likeliness, _ = self.neural_network.score_list(new_states)
            if self.scoring_function is not None:
                rewards = self.scoring_function.score_list(new_states)
                rewards = np.stack((np.array(rewards), drug_likeliness), axis=1)
                self.print_rewards = np.array(rewards[:, 0])
            else:
                rewards = drug_likeliness
                self.print_rewards = drug_likeliness
        else:
            if self.scoring_function is not None:
                rewards = self.scoring_function.score_list(new_states)
            else:
                rewards = np.zeros(self.n_walkers)
            self.print_rewards = np.array(rewards)
        self.observations = [molecules.get_fingerprint(state) for state in new_states]

        # rewards = np.stack((np.array(rewards), lengths), axis=1)

        self.times = (self.times + self.dt).astype(np.int32)

        # Calculate custom rewards and boundary conditions

        # Save data and update sample count
        new_ids = self._n_samples_done + np.arange(self.n_walkers).astype(int)
        self.walkers_id = new_ids
        self.data.append(walker_ids=new_ids, states=new_states, actions=new_states)

        if self.accumulate_rewards:
            self.rewards = self.rewards + np.array(rewards)
        else:
            self.rewards = np.array(rewards)
        # Stop all the walkers

        self._n_samples_done += self.n_walkers

    def evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """
        # Get random companion
        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))
        # Euclidean distance between states (pixels / RAM)
        dist = [
            DataStructs.DiceSimilarity(self.observations[i], self.observations[idx[i]])
            for i in range(self.n_walkers)
        ]

        dist = 1.0 - np.array(dist)

        return relativize_vector(dist).astype(np.float32)

    def normalize_rewards(self) -> np.ndarray:
        """We also apply the relativize function to the rewards"""
        if self.neural_network is not None and self.scoring_function is not None:
            rewards1 = relativize_vector(np.array(self.rewards[:, 0])).astype(np.float32)
            rewards2 = relativize_vector(np.array(self.rewards[:, 1])).astype(np.float32)
            normed_rewards = rewards1 * rewards2
        else:
            rewards = np.array(self.rewards)
            normed_rewards = relativize_vector(rewards).astype(np.float32)
        return normed_rewards

    def virtual_reward(self) -> np.ndarray:  # modifie
        """Calculate the virtual reward of the walkers. This quantity is used for determining
        the chance a given walker has of cloning. This scalar gives us a measure of how well a
        walker is solving the exploration vs exploitation problem, with respect to the other
        walkers of the Swarm.
        """
        dist = self.evaluate_distance()  # goes between 0 and 1
        scores = self.normalize_rewards()
        vir_reward = scores * (dist ** self.balance)
        self._virtual_reward = vir_reward

        return self._virtual_reward

    def get_clone_compas(self):  # modified

        alive_walkers = np.arange(self.n_walkers, dtype=int)
        self._clone_idx = np.random.choice(alive_walkers, self.n_walkers)

        return self._virtual_reward[self._clone_idx]

    def clone_condition(self):
        """Calculates the walkers that will cone depending on their virtual rewards. Returns the
        index of the random companion chosen for comparing virtual rewards.
        """

        self._pre_clone_ids = list(set(self.walkers_id.astype(int)))
        # Calculate virtual rewards and choose another walker at random
        vir_rew = self.virtual_reward()
        vr_compas = self.get_clone_compas()
        if vr_compas is None:
            self._will_clone = np.zeros(self.n_walkers, dtype=bool)
            return
        value = (vr_compas - vir_rew) / np.where(vir_rew > 0, vir_rew, 1e-8)
        self._will_clone = (value >= np.random.random()).astype(bool)

    def perform_clone(self):
        idx = self._clone_idx
        # A hack that avoid cloning
        if idx is None:
            return
        # This is a hack to make it work on n dimensional arrays
        # Using np.where seems to be faster than using a for loop

        rew_ix = self._will_clone[
            (...,) + tuple(np.newaxis for _ in range(len(self.rewards.shape) - 1))
        ]
        self.rewards = np.where(rew_ix, self.rewards[idx], self.rewards)
        self._virtual_reward = np.where(
            self._will_clone, self._virtual_reward[idx], self._virtual_reward
        )

        # self.times = np.where(self._will_clone, self.times[idx], self.times)
        # self.observations = np.where(self._will_clone, np.array(self.observations)[idx], np.array(self.observations))
        # self.observations = self.observations.tolist()

        # self.not_frozen = np.logical_and(self._will_clone, np.logical_not(self._will_clone[idx]))
        # self.not_frozen = np.where(self._will_clone, self.not_frozen[idx], self.not_frozen)

        self.walkers_id = np.where(self._will_clone, self.walkers_id[idx], self.walkers_id).astype(
            int
        )

    def update_data(self):  # modifed
        """Update the states and observations of the swarm kept in self.data."""
        self._post_clone_ids = set(self.walkers_id.astype(int))
        self.data.update_values(self._post_clone_ids)

    def clone(self):
        """The clone operator aims to change the distribution of walkers in the state space, by
         cloning some walkers to a randomly chosen companion. After cloning, the distribution of
         walkers will be closer to the reward distribution of the state space.
        1 - Choose a random companion who is alive.
        2 - Calculate the probability of cloning based on their virtual reward relationship.
        3 - Clone if p > random[0,1] or the walker is dead.
        """
        self.perform_clone()
        self.update_data()

    def run_swarm(self, iterations, state=None, reset=True, print_swarm: bool = False):
        """
        Iterate the swarm by either evolving or cloning each walker until a certain condition
        is met.
        :return:
        """

        if reset:
            self.init_swarm(state=state)
            self.all_data = {}

        for _ in tqdm(range(iterations)):
            if self._i_simulation > 1:
                self.clone_condition()
                self.clone()
            self.step_walkers()
            self._i_simulation += 1
            data = self.data.get_states(self.walkers_id)
            rewards = self.rewards
            for i in range(len(data)):
                if data[i] not in self.all_data:
                    self.all_data[data[i]] = rewards[i]

            if self._i_simulation % self.render_every == 0 and print_swarm:
                print(self)
                clear_output(True)

        if print_swarm:
            print(self)

        data = []
        rewards = []
        for key in self.all_data.keys():
            data.append(key)
            rewards.append(self.all_data[key])
        data = np.array(data)
        rewards = np.array(rewards)

        return data, rewards

    def run_swarm_training(self, number_data, state=None, reset=True, print_swarm: bool = False):
        """
        Iterate the swarm by either evolving or cloning each walker until a certain condition
        is met.
        :return:
        """

        if reset:
            self.init_swarm(state=state)
            self.all_data = {}

        while len(self.all_data) < number_data:
            if self._i_simulation > 1:
                self.clone_condition()
                self.clone()
            self.step_walkers()
            self._i_simulation += 1
            data = self.data.get_states(self.walkers_id)
            rewards = self.rewards
            for i in range(len(data)):
                if data[i] not in self.all_data:
                    self.all_data[data[i]] = rewards[i]

            if self._i_simulation % self.render_every == 0 and print_swarm:
                print(self)
                clear_output(True)

        if print_swarm:
            print(self)

        data = []
        rewards = []
        for key in self.all_data.keys():
            data.append(key)
            rewards.append(self.all_data[key])
        data = np.array(data)
        rewards = np.array(rewards)

        return data, rewards
