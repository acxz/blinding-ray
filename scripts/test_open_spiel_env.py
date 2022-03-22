import pyspiel

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.tune import register_env

from blinding_ray.agents.random import RandomTrainer

# RBC OpenSpiel env
register_env("open_spiel_env_rbc", lambda _: OpenSpielEnv(
    pyspiel.load_game("rbc")))
    #pyspiel.load_game("connect_four")))

# Configure the algorithm
config = {
    "env": "open_spiel_env_rbc",
    "model": {
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "relu",
    },
    "num_workers": 1,
    "framework": "torch",
}

# Create our RLlib trainer
#trainer = PPOTrainer(config=config)
trainer = RandomTrainer(config=config)

# Run training iterations
iterations = 1
for _ in range(iterations):
    print(trainer.train())