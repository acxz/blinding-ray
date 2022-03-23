# Based on the attacker implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/attacker_bot.py

import random
from typing import Dict, Tuple

import chess
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType


class AttackerPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # move sequences from white's perspective
        # flipped at runtime if playing as black
        quick_attacks = [
            # queen-side knight attacks
            [chess.Move(chess.B1, chess.C3), chess.Move(chess.C3, chess.B5),
             chess.Move(chess.B5, chess.D6), chess.Move(chess.D6, chess.E8)],
            [chess.Move(chess.B1, chess.C3), chess.Move(chess.C3, chess.E4),
             chess.Move(chess.E4, chess.F6), chess.Move(chess.F6, chess.E8)],

            # king-side knight attacks
            [chess.Move(chess.G1, chess.H3), chess.Move(chess.H3, chess.F4),
             chess.Move(chess.F4, chess.H5), chess.Move(chess.H5, chess.F6),
             chess.Move(chess.F6, chess.E8)],

            # four move mates
            [chess.Move(chess.E2, chess.E4), chess.Move(chess.F1, chess.C4),
             chess.Move(chess.D1, chess.H5), chess.Move(chess.C4, chess.F7),
             chess.Move(chess.F7, chess.E8), chess.Move(chess.H5, chess.E8)],
        ]

        self.move_sequence = random.choice(quick_attacks)

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
        # one env step it sense, the next is action, the next is sense, ...

        # See: https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/pybind11/pyspiel.cc
        # for pyspiel api

        # openspiel has no method for state.get_phase() to determine if
        # sensing or moving
        # Thus the state.legal_actions() is used to determine which phase
        # Could use the observation space directly, but hard to read from
        # sensing has action space from [0...35]
        # moving has action space from [0...4672] (env.num_distinct_actions())

        if info_batch[0] != 0:
            state = info_batch[0]['state']
            black = 0
            if state.current_player() == black:
                chess_move_sequence = list(
                    map(flipped_move, self.move_sequence))
            else:
                chess_move_sequence = self.move_sequence

            # TODO: move to render env
            board = chess.Board(state.__str__())
            #print(board)

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

# Maybe import this directly from reconchess and depend on it


def flipped_move(move):
    def flipped(square):
        return chess.square(chess.square_file(square),
                            7 - chess.square_rank(square))

    return chess.Move(from_square=flipped(move.from_square),
                      to_square=flipped(move.to_square),
                      promotion=move.promotion, drop=move.drop)
