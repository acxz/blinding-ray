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
    "env": "open_spiel_env_rbc",
    "num_workers": 1,
}

# Create our RLlib trainer
trainer = RandomTrainer(config=config)
# trainer = AttackerTrainer(config=config)

# Run training iterations
iterations = 1
for _ in range(iterations):
    print(trainer.train())
