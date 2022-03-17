""" ReconChessEnv """
# Create an external env to interface with reconchess's localgame class to play
# a reconchess game
# Extended from:
# https://github.com/ray-project/ray/blob/master/rllib/env/external_env.py
# and https://github.com/ray-project/ray/blob/
# 7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/rllib/tests/test_external_env.py
# make it into a cooperative agent since we have both the senser and the
# mover
# TODO: https://github.com/ray-project/ray/blob/master/rllib/env/external_multi_agent_env.py

from typing import Optional

from gym import spaces
import numpy as np
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.typing import (
    EnvActionType,
    EnvConfigDict,
    EnvInfoDict,
    EnvObsType,
)


class ReconChessEnv(ExternalEnv):
    """ ReconChessEnv """

    def __init__(
            self,
            env_config: EnvConfigDict):

        # TODO need to configure proper action/observation space
        action_high = np.array([np.finfo(np.float32).max,
                                np.finfo(np.float32).max], dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.pi,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        action_space = spaces.Box(-action_high,
                                  action_high, dtype=np.float32)
        observation_space = spaces.Box(-obs_high,
                                       obs_high, dtype=np.float32)

        ExternalEnv.__init__(self, action_space, observation_space)

        print("created reconchessenv\n")

    def run(self):
        print("You tryna run?\n")

    def start_episode(
            self, episode_id: Optional[str] = None,
            training_enabled: bool = True) -> str:

        print("Episode is in start.\n")

    def get_action(
            self,
            episode_id: str,
            observation: EnvObsType) -> EnvActionType:

        print("Getting Action.\n")

    def log_action(
            self, episode_id: str, observation: EnvObsType,
            action: EnvActionType) -> None:

        print("Logging Action.\n")

    def log_returns(self, episode_id: str, reward: float, info:
                    Optional[EnvInfoDict] = None) -> None:

        print("Loggin return.\n")

    def end_episode(
            self, episode_id: str, observation: EnvObsType) -> None:

        print("Ending episode.\n")

# generate & register ReconChessEnv class
# ReconChessEnv = make_reconchess_env()
