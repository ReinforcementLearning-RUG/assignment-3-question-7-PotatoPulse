import numpy as np
from mdp.abstract_mdp import AbstractMDP
from model_free_prediction.abstract_evaluator import AbstractEvaluator
from policy.abstract_policy import AbstractPolicy


class TDEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float):
        """
        Initializes the TD(0) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        """
        self.env = env
        self.alpha = alpha
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)              # Reset value function.

        for _ in range(num_episodes):
            self._update_value_function(policy)

        return self.value_fun.copy()

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(0) method to update the value function.
        :param policy: A policy object that provides action probabilities for each state.
        """
        state = self.env.reset()
        
        while True:
            action = policy.sample_action(state)
            
            next_state, reward, done = self.env.step(action)
            
            self.value_fun[state] += self.alpha * (reward + 
                                     (self.env.discount_factor * self.value_fun[next_state] if not done else 0) 
                                     - self.value_fun[state])
            
            state = next_state.copy()
            
            if done:
                break