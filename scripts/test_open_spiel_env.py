import pyspiel
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.tune import register_env

from blinding_ray.agents.attacker import AttackerTrainer
from blinding_ray.agents.random import RandomTrainer

# RBC OpenSpiel env
register_env("open_spiel_env_rbc", lambda _: OpenSpielEnv(
    pyspiel.load_game("rbc")))

# Configure the algorithm
config = {
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "env": "open_spiel_env_rbc",
    "render_env": False,
    "log_sys_usage": True,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": True, # This does not work if record_env is False
        "record_env": True,
    },
    "num_gpus": 0,
    "output": "logdir",
    "output_compress_columns": ["obs", "new_obs"],
    "multiagent": {
       "policies": {
           # Figure this out, prob the answer to know our opponent
       }
    }
}

# Create our RLlib trainer
trainer = RandomTrainer(config=config)
# trainer = AttackerTrainer(config=config)

# Run training iterations
#print("Training")
#iterations = 1
#for _ in range(iterations):
#    print(trainer.train())

# Run evaluation
print("Evaluating")
trainer.evaluate()