from typing import List, Tuple, Optional
import numpy as np
from mdp.abstract_mdp import AbstractMDP
from mdp.reward_function import RewardFunction
from mdp.transition_function import TransitionFunction


class MDP(AbstractMDP):
    def __init__(
            self,
            states: List[int],
            actions: List[int],
            transition_function: TransitionFunction,
            reward_function: RewardFunction,
            discount_factor: float = 0.9,
            terminal_state: Optional[int] = None,
            start_state: Optional[int] = 0
    ):
        """
        Initializes the Markov Decision Process (MDP).

        :param states: A list of states in the MDP.
        :param actions: A list of actions in the MDP.
        :param transition_function: A TransitionFunction object that provides transition probabilities.
        :param reward_function: A RewardFunction object that provides rewards.
        :param discount_factor: A discount factor for future rewards.
        :param terminal_state: A terminal state.
        :param start_state: A starting state. If set, then reset() will always return that state.
        """
        self._states = states
        self._actions = actions
        self._transition_function = transition_function
        self._reward_function = reward_function
        self._discount_factor = discount_factor

        self._start_state = start_state
        self._curr_state = self._start_state if self._start_state is not None else np.random.choice(self._states)
        self._terminal_state = terminal_state       # Assuming one terminal state for simplicity.

    def reset(self) -> int:
        """
        Re-initialize the state by sampling uniformly from the state space.
        :return: New initial state.
        """
        self._curr_state = self._start_state if self._start_state is not None else np.random.choice(self._states)
        return self._curr_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Perform a realization of p(s'|s,a) and r(s,a).

        :param action: Action taken by the agent.

        :return: A tuple containing the new state, the reward, and a done flag.
        """
        # Get the transition probabilities for the current state and action.
        transition_probs = self._transition_function(self._curr_state, action)

        # Sample the next state based on the transition probabilities.
        next_state = np.random.choice(self._states, p=transition_probs)

        # Calculate the reward for the current state and action.
        reward = self._reward_function(self._curr_state, action)

        self._curr_state = next_state

        done = False if self._terminal_state is None else next_state == self._terminal_state

        return next_state, reward, done

    def transition_prob(self, new_state: int, state: int, action: int) -> float | np.ndarray:
        """
        Returns the transition probabilities for the new state given state and action by calling the transition
        function.

        :param new_state: New state
        :param state: Current state
        :param action: Action taken

        :return: Probability p(s'|s,a).
        """
        return self._transition_function(state, action)[new_state]

    def reward(self, state: int, action: int) -> float:
        """
        Returns the reward for a given state and action by calling the reward function.

        :param state: Current state
        :param action: Action taken

        :return: A float representing the reward for the given (state, action) pair.
        """
        return self._reward_function(state, action)

    @property
    def states(self) -> List[int]:
        """
        Returns the list of states.

        :return: A list of states.
        """
        return self._states

    @property
    def actions(self) -> List[int]:
        """
        Returns the list of actions.

        :return: A list of actions.
        """
        return self._actions

    @property
    def discount_factor(self) -> float:
        """
        Returns the discount factor.

        :return: The discount factor.
        """
        return self._discount_factor

    @property
    def num_states(self) -> int:
        """
        :return: The number of states.
        """
        return len(self._states)

    @property
    def num_actions(self) -> int:
        """
        :return: The number of actions.
        """
        return len(self._actions)

    @property
    def current_state(self) -> int:
        """
        :return: The current state.
        """
        return self._curr_state
