import random

import pyspiel
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.trainer import Trainer
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.tune import register_env

from blinding_ray.agents.attacker import AttackerPolicy
from blinding_ray.agents.random import RandomPolicy
from blinding_ray.agents.trout import TroutPolicy

# RBC OpenSpiel env
register_env("open_spiel_env_rbc", lambda _: OpenSpielEnv(
    pyspiel.load_game("rbc")))

# Policy Mapping from Agent ID to Policy ID
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # Choose a random side for player and opponent
    # agent_id = 1: white
    # agent_id = 0: black
    colors = [1, 0]
    player_color = random.choice(colors)
    if (agent_id == player_color):
        policy_id = "player"
    else:
        policy_id = "opponent"

    return policy_id

# Configure the algorithm
config = {
    "num_workers": 0, # num of cpus to use minus one
    "num_envs_per_worker": 1,
    "batch_mode": "complete_episodes",
    "env": "open_spiel_env_rbc",
    "render_env": False,
    "log_sys_usage": True,
    "evaluation_num_workers": 1,
    "evaluation_duration": 1, # Play for one episode
    "evaluation_config": {
        "render_env": True, # This does not work if record_env is False
        "record_env": True, # Is this even recording?
    },
    "num_gpus": 0,
    "output": "logdir",
    "output_compress_columns": ["obs", "new_obs"],
    "multiagent": {
       "policies": {
           "player": PolicySpec(policy_class=AttackerPolicy),
           "opponent": PolicySpec(policy_class=RandomPolicy),
       },
       "policy_mapping_fn": policy_mapping_fn,
       "policies_to_train": ["player"],
    },
    #"callbacks": RandomCallbacks,
}

# Create our RLlib trainer
trainer = Trainer(config=config)

# Run training iterations
print("Training")
iterations = 1
for _ in range(iterations):
    print(trainer.train())

# Run evaluation
#print("Evaluating")
#trainer.evaluate()