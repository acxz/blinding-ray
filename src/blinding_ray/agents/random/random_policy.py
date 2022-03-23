# Based on the attacker implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/random_bot.py

from typing import Dict, Tuple

import numpy as np
from gym.spaces import Box
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType


class RandomPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # Pick a random legal move from state.legal_actions()
        if info_batch[0] != 0:
            state = info_batch[0]['state']
            legal_actions = state.legal_actions()
            actions = [np.random.choice(legal_actions) for _ in obs_batch]
        else:
            # TODO still need infos at the first state
            actions = [self.action_space.sample()
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
