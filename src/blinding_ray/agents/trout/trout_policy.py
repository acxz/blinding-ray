# Based on the trout implementation bot from reconchess.
# See: https://github.com/reconnaissanceblindchess/reconchess/blob/master/reconchess/bots/trout_bot.py

import copy
from typing import Dict, Tuple

import chess
import numpy as np
import pyspiel
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType


class TroutPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.engine = None

        # This is a hack needed to obtain functionality of action_to_string
        # while the phase is move instead of sense and vice versa
        # See: https://github.com/deepmind/open_spiel/blob/062416fdd173424440acfa8938867df8b9e22735/open_spiel/games/rbc.cc#L444-L454
        # Note prev_state needs to be set before any return statement
        # BUG: action_to_string and string_to_action should work even if moves
        # are not legal_actions at the current state, they should also take the
        # current phase (sense/move) and use that instead of depending on the
        # phase from the state object
        self.prev_state = None
        self.first_move = True

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
        # MAYBE: need to use state_batches instead of self for persistant data
        # across the policy do I tho?

        # TODO: This entire method needs to be batched
        # for now deal with batch=1
        if info_batch[0] != 0:
            state = info_batch[0]['state']
            legal_actions = state.legal_actions()
            state_board = chess.Board(state.__str__())

            # Parse observation string
            observed_board, castling_rights, phase, capture, color, \
                illegal_move = parse_observation_string(state)

            # (s)ensing phase
            if phase == 's':

                # BUG: open_spiel records the moves commanded in state.history()
                # not neccessarily the ones taken (i.e. if opt piece is in the
                # motion of your bishop)
                # Maybe take a look at exposing MovesHistory in open_spiel

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
                    taken_action)
                # if taken action was pass then don't push to board
                if taken_action_string != 'pass':
                    taken_move = chess.Move.from_uci(taken_action_string)
                    self.board.push(taken_move)

                # handle_opponent_move_result
                # determine captured square
                opponent_taken_action = state.history()[-1]
                # if opponent passes just skip this
                if opponent_taken_action != 0:
                    opponent_taken_action_string = \
                        self.prev_state.action_to_string(opponent_taken_action)
                    opponent_taken_move = \
                        chess.Move.from_uci(opponent_taken_action_string)
                    opponent_capture_square = \
                        chess.Square(opponent_taken_move.to_square)
                    opponent_capture_square_file = \
                        chess.square_name(opponent_capture_square)[0]
                    opponent_capture_square_rank = \
                        chess.square_name(opponent_capture_square)[-1]
                    capture_square_file = opponent_capture_square_file
                    # Flip the opponent_capture_square to match our side
                    # Look at chess.square_mirror
                    capture_square_rank = \
                        str(9 - int(opponent_capture_square_rank))
                    capture_square = chess.parse_square(capture_square_file +
                                                        capture_square_rank)
                    captured_piece = \
                        self.board.piece_at(capture_square)
                    # make sure the captured piece is one of ours and if so
                    # remove our piece from our internal board
                    my_piece = None
                    if (captured_piece is not None and
                            captured_piece.color == self.color):
                        my_piece = captured_piece
                    captured_my_piece = my_piece is not None
                    if captured_my_piece:
                        self.my_piece_captured_square = \
                            capture_square
                        self.board.remove_piece_at(capture_square)
                    else:
                        self.my_piece_captured_square = None

                # choose_sense
                # if our piece was just captured, sense where it was captured
                if self.my_piece_captured_square:
                    open_spiel_action = convert_sense_action(
                        state,
                        self.my_piece_captured_square)
                    actions = [open_spiel_action for _ in
                               obs_batch]

                    # retain copy of previous state to use for action to string
                    self.prev_state = copy.copy(state)
                    return actions, [], {}

                # if we might capture a piece when we move, sense where the capture will occur
                future_move = self.choose_move()
                if (future_move is not None and
                        self.board.piece_at(future_move.to_square)
                        is not None):
                    open_spiel_action = convert_sense_action(
                        state,
                        future_move.to_square)
                    if open_spiel_action in legal_actions:
                        actions = [open_spiel_action for _ in obs_batch]
                    else:
                        # random fallback needed to due a bug in
                        # convert_sense_action
                        actions = [np.random.choice(legal_actions)
                                   for _ in obs_batch]

                    # retain copy of previous state to use for action to string
                    self.prev_state = copy.copy(state)

                    return actions, [], {}

                # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
                sense_actions = [*chess.SQUARES]
                for square, piece in self.board.piece_map().items():
                    if piece.color == self.color:
                        sense_actions.remove(square)
                # convert sense actions to actions in open_spiel format
                open_spiel_actions = []
                for sense_action in sense_actions:
                    open_spiel_action = convert_sense_action(state,
                                                             sense_action)
                    if open_spiel_action in legal_actions:
                        open_spiel_actions.append(open_spiel_action)
                actions = [np.random.choice(open_spiel_actions)
                           for _ in obs_batch]

                # retain copy of previous state to use for action to string
                self.prev_state = copy.copy(state)

                return actions, [], {}
            else:
                # (m)oving phase

                # handle_sense_result
                # add the pieces in the sense result to our board
                # observed_board variable keeps track of sensed results
                # BUG it does not start with a prior on the opponent pieces
                # i.e. there starting locations, thus we use the results
                # directly from state and apply the sense results ourselves to
                # self.board
                # open_spiel upstream issue

                prev_sense_action = state.history()[-1]
                # dont have access to self.prev_state in the first move
                # ISSUE: unfair advantage to white as white doesn't sense anything
                # anyway
                prev_sense_square = 'b2'  # dummy sense square for first move
                if self.first_move:
                    self.first_move = False
                else:
                    prev_sense_square = \
                        self.prev_state.action_to_string(
                            prev_sense_action)[-2:None]

                sensed_squares = get_sense_grid(prev_sense_square)

                for sensed_square in sensed_squares:
                    sensed_piece = state_board.piece_at(sensed_square)
                    self.board.set_piece_at(sensed_square, sensed_piece)

                # choose_move
                move_action = self.choose_move()
                action = convert_move_action(state, move_action)
                actions = [action for _ in obs_batch]

                # retain copy of previous state to use for action to string
                self.prev_state = copy.copy(state)

                return actions, [], {}
        else:
            # TODO still need infos at the first step/obs
            # if self.color == chess.WHITE this is fine, since there is nothing
            # to sense
            # if self.color == chess.BLACK then we should:
            # handle_move_result
            # we didn't move yet so this is also fine
            # handle_opponent_move_result
            # for the first move of the game, no piece can be taken anyway so
            # no need to handle
            # choose_sense
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

    def choose_move(self):
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
                return attack_action
        # otherwise, try to move with the stockfish chess engine
        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(
                self.board, chess.engine.Limit(time=0.5))
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(
                self.board.fen()))

        # if all else fails, pass
        return None


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


def convert_sense_action(state, sense_action):
    sense_string = "Sense " + chess.square_name(sense_action)

    # BUG: open_spiel has a bug on the mapping of sense actions
    # it tries to reduce sense space with the inner game board
    # 6x6 but fails to account shift the squares to be in the
    # middle 6x6
    # See: https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/rbc.cc#L423
    # Thus we try/except and filter from legal_actions
    try:
        open_spiel_action = state.string_to_action(
            sense_string)
        return open_spiel_action
    except (pyspiel.SpielError) as e:
        return None


def convert_move_action(state, move_action):
    # convert move action to action space actions
    # value of 0 corresponds to pass
    open_spiel_action = 0
    if move_action is not None:
        move_action_string = move_action.uci()
        # RBC requires pawn promotion to queen
        # if pawn is moving to rank 8 or rank 1
        if move_action_string[-1] == 8 or move_action_string[-1] == 1:
            # Check if piece is pawn
            chess_board = chess.Board(state.__str__())
            if (chess_board.piece_type_at(move_action.from_square) ==
                    chess.PAWN):
                move_action_string = move_action_string + 'q'
        # Remember since stockfish does not have access to the full board state
        # it can try to perform an illegal action.
        # If so, pass for the current move
        open_spiel_action = 0
        try:
            open_spiel_action = state.string_to_action(move_action_string)
        except pyspiel.SpielError as e:
            print("Stockfish attempting an illegal move")

    return open_spiel_action


def get_sense_grid(sense_square):
    file = sense_square[0]
    rank = sense_square[1]
    sense_grid = []
    file_min_offset = -1
    file_max_offset = 1
    rank_min_offset = -1
    rank_max_offset = 1
    if file == 'a':
        file_min_offset = 0
    if file == 'h':
        file_max_offset = 0
    if rank == '1':
        rank_min_offset = 0
    if rank == '8':
        rank_max_offset = 0
    for file_offset in range(file_min_offset, file_max_offset+1):
        new_file = chr(ord(file) + file_offset)
        for rank_offset in range(rank_min_offset, rank_max_offset+1):
            new_rank = str(int(rank) + rank_offset)
            new_square = new_file + new_rank
            sense_grid.append(chess.parse_square(new_square))

    return sense_grid
