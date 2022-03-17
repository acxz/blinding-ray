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
import numpy as np
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    EnvConfigDict,
    MultiAgentDict,
)


class ReconChessEnv(ExternalMultiAgentEnv):
    """ ReconChessEnv """

    def __init__(
            self,
            env_config: EnvConfigDict):

        # LOOK need to configure proper action/observation space
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

        ExternalMultiAgentEnv.__init__(self, action_space, observation_space)

        print("created reconchessenv")

    def run(self):
        print("You tryna run?")
        # Think we gotta import the localgame from reconchess in here or
        # playlocalgame

        episode_id = self.start_episode()
        print(episode_id)

    @override(ExternalMultiAgentEnv)
    def start_episode(
            self,
            episode_id: Optional[str] = None,
            training_enabled: bool = True,
    ) -> str:

        print("Episode is in start.")

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
