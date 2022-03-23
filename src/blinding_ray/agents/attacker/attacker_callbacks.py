# Callbacks that the Attacker Policy requires

import random
from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from reconchess.bots.attacker_bot import QUICK_ATTACKS


class AttackerCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy], episode: Episode,
                         **kwargs) -> None:

        # Choose a different attack move sequence
        policies['attacker'].move_sequence = random.choice(QUICK_ATTACKS)
