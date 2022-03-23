# Based on the attacker implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/attacker_bot.py
# TODO: Need to implement, right now just shell of RandomPolicy

import random
from typing import Dict, Tuple

import numpy as np
from gym.spaces import Box
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType


class AttackerPolicy(Policy):

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
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs
    ):
        # one env step it sense, the next is action, the next is sense, ...

        # See: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/pybind11/pyspiel.cc
        # for pyspiel api

        # openspiel has no method for state.get_phase() to determine if
        # sensing or moving
        # Thus the state.legal_actions() is used to determine which phase
        # Could use the observation space directly, but hard to read from
        # sensing has action space from [0...35]
        # moving has action space from [0...4672] (env.num_distinct_actions())

        # Pick a random legal move from state.legal_actions()
        if info_batch[0] != 0:
            legal_actions = info_batch[0]['state'].legal_actions()
            actions = [np.random.choice(legal_actions) for _ in obs_batch]
        else:
            # TODO still need infos at the first state
            actions = [self.action_space_for_sampling.sample()
                       for _ in obs_batch]

        return actions, [], {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        "No log likelihoods to compute."""
        return np.array([0] * len(obs_batch))

    # not implemented loss

    @override(Policy)
    def load_batch_into_buffer(self, batch: SampleBatch,
                               buffer_index: int = 0) -> int:
        """No buffers to load batch into."""
        return 0

    # not implemented get_num_samples_loaded_into_buffer

    @override(Policy)
    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        """No loaded batches to learn from."""
        return [0]

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradients to compute."""
        return None, None

    def apply_gradients(self, gradients: ModelGradients) -> None:
        """No gradients to apply."""
        pass

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    # not implemented export_checkpoint

    # not implemented import_model_from_h5
