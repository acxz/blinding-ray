""" ReconChessEnv """
# Create an external env to interface with reconchess's localgame class to play
# a reconchess game
# Extended from:
# https://github.com/ray-project/ray/blob/master/rllib/env/external_env.py
# and https://github.com/ray-project/ray/blob/
# 7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/rllib/tests/test_external_env.py
# make it into a cooperative agent since we have both the senser and the
# mover
# LOOK:
# https://github.com/ray-project/ray/blob/master/rllib/env/external_multi_agent_env.py
# LOOK: maybe only need to override run

from typing import Optional

from gym import spaces
from gym.spaces import Tuple
import numpy as np
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    EnvConfigDict,
    MultiAgentDict,
)

import chess
from reconchess import load_player, LocalGame, Player
# from reconchess.player import Player


class ReconChessEnv(ExternalMultiAgentEnv):
    """ ReconChessEnv """

    def __init__(
            self,
            env_config: EnvConfigDict):

        import pprint
        pprint.pprint(env_config)
        # TODO wrap around if statement
        self.seconds_per_player = 900  # Default of reconchess
        self.white_bot_path = "white"
        self.black_bot_path = "black"

        # TODO need to configure proper action/observation space
        action_high = np.array([np.finfo(np.float32).max,
                               np.finfo(np.float32).max], dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                            np.finfo(np.float32).max,
                            np.pi,
                            np.finfo(np.float32).max],
                            dtype=np.float32)

        action_space_og = spaces.Box(-action_high,
                                     action_high, dtype=np.float32)
        observation_space_og = spaces.Box(-obs_high,
                                          obs_high, dtype=np.float32)

        # Two agents: one for making a sense another for making a move
        self.agent_sense = 0
        self.agent_move = 0

        observation_space_sense = observation_space_og
        observation_space_move = observation_space_og

        action_space_sense = action_space_og
        action_space_move = action_space_og

        observation_space = Tuple(
            [observation_space_sense,
             observation_space_move])

        action_space = Tuple(
            [action_space_sense,
             action_space_move])

        ExternalMultiAgentEnv.__init__(self, action_space, observation_space)

        print("created reconchessenv")

    def run(self):
        print("You tryna run?")
        # Think we gotta import the localgame from reconchess in here or
        # playlocalgame

        _, white_player_cls = load_player(self.white_bot_path)
        _, black_player_cls = load_player(self.black_bot_path)

        self.game = LocalGame(self.seconds_per_player)

        episode_id = self.start_episode(white_player_cls(), black_player_cls(),
                game=game)
        print(episode_id)

        while True:
            # reconchess/play.py
            # play_turn(game, players[game.turn], end_turn_last=True)

            # Your loop should continuously:
            # 1. Call self.start_episode(episode_id)
            # 2. Call self.get_action(episode_id, obs_dict)
            #         -or-
            #         self.log_action(episode_id, obs_dict, action_dict)
            # 3. Call self.log_returns(episode_id, reward_dict)
            # 4. Call self.end_episode(episode_id, obs_dict)
            # 5. Wait if nothing to do.
            #print("game loop")

    @override(ExternalMultiAgentEnv)
    def start_episode(
            self,
            white_player: Player,
            black_player: Player,
            seconds_per_player: float = 900,
            episode_id: Optional[str] = None,
            training_enabled: bool = True,
    ) -> str:

        print("Episode is in start.")
        # See reconchess/play.py/play_local_game


        self.game = game
        self.players = [black_player, white_player]

        white_name = white_player.__class__.__name__
        black_name = black_player.__class__.__name__
        self.game.store_players(white_name, black_name)

        white_player.handle_game_start(chess.WHITE, game.board.copy(),
                                       black_name)
        black_player.handle_game_start(chess.BLACK, game.board.copy(),
                                       white_name)

        self.game.start()

        ExternalMultiAgentEnv.start_episode(self, episode_id, training_enabled)

    @override(ExternalMultiAgentEnv)
    def get_action(
            self,
            episode_id: str,
            observation_dict: MultiAgentDict,
    ) -> MultiAgentDict:

        print("Getting Action.")

        ExternalMultiAgentEnv.get_action(self, episode_id, observation_dict)

    @override(ExternalMultiAgentEnv)
    def log_action(
        self,
        episode_id: str,
        observation_dict: MultiAgentDict,
        action_dict: MultiAgentDict,
    ) -> None:

        print("Logging Action.")

        ExternalMultiAgentEnv.log_action(self, episode_id, observation_dict,
                                         action_dict)

    @override(ExternalMultiAgentEnv)
    def log_returns(
            self,
            episode_id: str,
            reward_dict: MultiAgentDict,
            info_dict: MultiAgentDict = None,
            multiagent_done_dict: MultiAgentDict = None,
    ) -> None:

        print("Loggin return.")

        ExternalMultiAgentEnv.log_returns(self, episode_id, reward_dict,
                                          info_dict, multiagent_done_dict)

    @override(ExternalMultiAgentEnv)
    def end_episode(
            self,
            episode_id: str,
            observation_dict: MultiAgentDict,
    ) -> None:

        print("Ending episode.")

        ExternalMultiAgentEnv.end_episode(self, episode_id, observation_dict)
