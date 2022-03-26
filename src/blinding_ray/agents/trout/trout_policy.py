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

        # This is a hack needed to obtain functionality of action_to_string
        # while the phase is move, instead of sense
        # See: https://github.com/deepmind/open_spiel/blob/062416fdd173424440acfa8938867df8b9e22735/open_spiel/games/rbc.cc#L444-L454
        self.prev_state = None

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
        # TODO: need to use state_batches instead of self for persistant data
        # across the policy do I tho?

        # TODO: This entire method needs to be batched
        # for now deal with batch=1
        if info_batch[0] != 0:
            state = info_batch[0]['state']

            # Parse observation string
            observed_board, castling_rights, phase, capture, color, \
                illegal_move = parse_observation_string(state)

            # (s)ensing phase
            if phase == 's':

                # TODO print that should be in env
                print(chess.Board(state.__str__()))

                # handle_move_result
                # if a move was executed, apply it to our board
                taken_action_history = state.history()
                color_shift = not self.color
                player_taken_action_history = \
                    [player_taken_action
                     for move_count in range(len(taken_action_history)//4)
                     for player_taken_action in
                     (taken_action_history[2*(2*move_count+color_shift)],
                      taken_action_history[2*(2*move_count+color_shift)+1])]
                taken_action = player_taken_action_history[-1]
                # transform action into string
                taken_action_string = self.prev_state.action_to_string(
                    self.color, taken_action)
                taken_move = chess.Move.from_uci(taken_action_string)
                if taken_move is not None:
                    self.board.push(taken_move)

                # TODO
                # handle opponent_move_result
                # Determine captured square
                # self.my_piece_captured_square = capture_square
                # if captured_my_piece:
                #    self.board.remove_piece_at(capture_square)

                # TODO
                # choose_sense
                # if our piece was just captured, sense where it was captured
                # if self.my_piece_captured_square:
                #     return self.my_piece_captured_square

                # # if we might capture a piece when we move, sense where the capture will occur
                # future_move = self.choose_move(move_actions, seconds_left)
                # if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
                #     return future_move.to_square

                # # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
                # for square, piece in self.board.piece_map().items():
                #     if piece.color == self.color:
                #         sense_actions.remove(square)
                # return random.choice(sense_actions)
                # Pick a random legal move for sensing
                legal_actions = state.legal_actions()
                actions = [np.random.choice(legal_actions) for _ in obs_batch]

                return actions, [], {}
            else:
                # (m)oving phase

                # retain copy of previous state to use for action to string
                import copy
                self.prev_state = copy.copy(state)

                # string_to_action may need self.prev_state if so need to move
                # this line right before the return statement

                # handle_sense_result
                # add the pieces in the sense result to our board
                # already handled in the observed_board variable

                # choose_move
                # if we might be able to take the king, try to
                enemy_king_square = self.board.king(not self.color)
                if enemy_king_square:
                    # if there are any ally pieces that can take king, execute
                    # one of those moves
                    enemy_king_attackers = self.board.attackers(
                        self.color, enemy_king_square)
                    if enemy_king_attackers:
                        attacker_square = enemy_king_attackers.pop()
                        attack_action = chess.Move(
                            attacker_square, enemy_king_square)
                        # convert attack_action to action space actions
                        attack_action_string = attack_action.uci()
                        # TODO make this a helper method
                        action = state.string_to_action(
                            self.color, attack_action_string)
                        actions = [action for _ in obs_batch]
                        return actions, [], {}
                # otherwise, try to move with the stockfish chess engine
                try:
                    self.board.turn = self.color
                    self.board.clear_stack()
                    result = self.engine.play(
                        self.board, chess.engine.Limit(time=0.5))
                    engine_action_string = result.move.uci()
                    action = state.string_to_action(
                        self.color, engine_action_string)
                    actions = [action for _ in obs_batch]
                    return actions, [], {}
                except chess.engine.EngineTerminatedError:
                    print('Stockfish Engine died')
                except chess.engine.EngineError:
                    print('Stockfish Engine bad state at "{}"'.format(
                        self.board.fen()))

            # if all else fails, pass
            # action value of 0 corresponds to pass
            actions = [0 for _ in obs_batch]

            # TODO: fix up logic for return points
            return actions, [], {}

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
