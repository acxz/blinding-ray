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

        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.engine = None

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

            # Parse observation string
            observed_board, castling_rights, phase, capture, color, \
                illegal_move = parse_observation_string(state)

            # TODO: handle opponent_move_result
            #self.my_piece_captured_square = capture_square
            # if captured_my_piece:
            #    self.board.remove_piece_at(capture_square)
            # choose_sense
            # handle_sense_result
            # choose_move
            # handle_move_result

            # Legal actions that can be taken this turn
            legal_actions = state.legal_actions()

            if legal_actions == list(range(0, 36)):
                phase = 'sensing'
            else:
                phase = 'moving'

            if phase == 'sensing':
                # Pick a random legal move for sensing
                actions = [np.random.choice(legal_actions) for _ in obs_batch]
            else:
                # pass... we failed so give up
                # action value of 0 corresponds to pass
                actions = [0 for _ in obs_batch]
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


def parse_observation_string(state):
    # See: https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/rbc.cc#L164

    # observed board
    obs_str = state.observation_string()
    split_slash_obs_str = obs_str.split('/')
    partial_observed_board = '/'.join(
        split_slash_obs_str[0:-1])
    last_slash_obs_str = split_slash_obs_str[-1]
    remaining_obs_str = last_slash_obs_str
    last_board_row = ""
    board_cols = 8
    last_row_count = 0
    for char in last_slash_obs_str:
        if last_row_count < board_cols:
            last_board_row = last_board_row + char
            remaining_obs_str = remaining_obs_str[1:None]
        if char.isdigit():
            last_row_count = last_row_count + int(char)
        else:
            last_row_count = last_row_count + 1
    observed_board = '/'.join([partial_observed_board, last_board_row])

    # castling rights
    split_space_rem_obs_str = remaining_obs_str.split(' ')[1:None]
    castling_rights = split_space_rem_obs_str[0]

    # phase
    phase = split_space_rem_obs_str[1]

    # capture
    capture = split_space_rem_obs_str[2]

    # color
    color = split_space_rem_obs_str[3]

    # illegal move
    illegal_move = split_space_rem_obs_str[4]

    return observed_board, castling_rights, phase, capture, color, illegal_move
