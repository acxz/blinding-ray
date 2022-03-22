import random
import numpy as np
from gym.spaces import Box
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights


class RandomPolicy(Policy):

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
        # sensing has action space from [0...35]
        # moving has action space from [0...4672] (env.num_distinct_actions())
        print(info_batch)
        # How do I populate info_batch?
        import pdb
        pdb.set_trace()

        # Need to sample from the proper action space
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        return [self.action_space_for_sampling.sample() for _ in obs_batch], [], {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        # TODO: Make this empty or 0s
        return np.array([random.random()] * len(obs_batch))

    # not implemented loss

    @override(Policy)
    def load_batch_into_buffer(self, batch: SampleBatch,
                               buffer_index: int = 0) -> int:
        print("am i tryna load")
        import pdb
        pdb.set_trace()
        return 0

    # not implemented get_num_samples_loaded_into_buffer

    @override(Policy)
    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        return [0]

    # not implemented compute_gradients

    # not implemented apply_gradients

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
