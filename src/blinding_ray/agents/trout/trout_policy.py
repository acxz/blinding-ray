# Based on the trout implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/trout_bot.py

from typing import Dict, Tuple

import chess
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType
from reconchess.bots.attacker_bot import flipped_move


class TroutPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_sequence = 0

    @ override(Policy)
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
        if info_batch[0] != 0:
            state = info_batch[0]['state']
            black = 0
            if state.current_player() == black:
                chess_move_sequence = list(
                    map(flipped_move, self.move_sequence))
            else:
                chess_move_sequence = self.move_sequence

            # Legal actions that can be taken this turn
            legal_actions = state.legal_actions()
            legal_action_strings = \
                [state.action_to_string(state.current_player(), legal_action)
                 for legal_action in legal_actions]

            # Convert move_sequence to action space actions
            move_sequence = []
            for move in chess_move_sequence:
                move_string = move.uci()
                if move_string in legal_action_strings:
                    openspiel_action = state.string_to_action(
                        state.current_player(), move_string)
                    move_sequence.append(openspiel_action)

            if legal_actions == list(range(0, 36)):
                phase = 'sensing'
            else:
                phase = 'moving'

            if phase == 'sensing':
                # Pick a random legal move for sensing
                actions = [np.random.choice(legal_actions) for _ in obs_batch]
            else:
                while (len(move_sequence) > 0 and
                        move_sequence[0] not in legal_actions):
                    move_sequence.pop(0)

                if len(move_sequence) == 0:
                    # pass... we failed so give up
                    # action value of 0 corresponds to pass
                    actions = [0 for _ in obs_batch]
                else:
                    # umm gotta pop for all obs_batch?
                    actions = [move_sequence.pop(0)]
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
