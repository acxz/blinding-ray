# Based on the attacker implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/attacker_bot.py
# TODO: Need to implement, right now just shell of RandomPolicy

import random

import numpy as np
from gym.spaces import Box
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights


class AttackerPolicy(Policy):
    """Hand-coded attacker policy that returns actions directly targeting
    king."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(
            self.action_space, Box
        ):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        **kwargs
    ):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        return [self.action_space_for_sampling.sample() for _ in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def load_batch_into_buffer(self, batch: SampleBatch,
                               buffer_index: int = 0) -> int:
        return 0

    @override(Policy)
    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        return [0]
