import pyspiel
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.agents.trainer import Trainer
from ray.tune import register_env

from blinding_ray.agents.attacker import AttackerCallbacks, AttackerPolicy
from blinding_ray.agents.random import RandomPolicy
from blinding_ray.agents.trout import TroutCallbacks, TroutPolicy
from blinding_ray.env.open_spiel_rbc import OpenSpielRbcEnv

# RBC OpenSpiel env
register_env("open_spiel_rbc_env", lambda _: OpenSpielRbcEnv(
    pyspiel.load_game("rbc")))


def policy_mapping_fn(agent_id, episode, worker, player="trout",
                      opponent="random", **kwargs):
    # TODO: Playing trout vs trout may not work as the policiesID will overlap
    # Need to use policy state instead of just the self to prevent this issue
    # Policy Mapping from Agent ID to Policy ID
    # player and opponent are the two policies that play against each other
    # See the policies dict in trainer config for what values these can be
    # agent_id = 1: white
    # agent_id = 0: black
    # Choose a side for player and opponent based on the episode_id
    player_color = episode.episode_id % 2
    if agent_id == player_color:
        policy_id = player
    else:
        policy_id = opponent

    return policy_id


# Configure the algorithm
config = {
    "num_workers": 0,  # num of cpus to use minus one
    "num_envs_per_worker": 1,
    "batch_mode": "complete_episodes",
    "env": "open_spiel_rbc_env",
    "render_env": False,
    "log_sys_usage": True,
    "evaluation_num_workers": 1,
    "evaluation_duration": 1,  # Play for one episode
    "evaluation_config": {
        "render_env": True,  # This does not work if record_env is False
        "record_env": True,  # Is this even recording?
    },
    "num_gpus": 0,
    "output": "logdir",
    "output_compress_columns": ["obs", "new_obs"],
    "multiagent": {
        "policies": {
            "attacker": PolicySpec(policy_class=AttackerPolicy),
            "random": PolicySpec(policy_class=RandomPolicy),
            "trout": PolicySpec(policy_class=TroutPolicy),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,
    },
    "callbacks": MultiCallbacks([
        AttackerCallbacks,
        TroutCallbacks,
    ]),
}

# Create our RLlib trainer
trainer = Trainer(config=config)

# Run training iterations
print("Training")
ITERATIONS = 2
for _ in range(ITERATIONS):
    trainer.train()

# Run evaluation
#print("Evaluating")
#EVAL_ITERATIONS = 1
#for _ in range(EVAL_ITERATIONS):
#    trainer.evaluate()
