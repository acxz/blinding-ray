# From https://github.com/ray-project/ray/blob/master/rllib/contrib/random_agent/random_agent.py
# created since ray.rllib.contrib.random_agent does not exist in shipped wheels
# upstream issue?
# Also it should probably just build off RandomPolicy

from ray.rllib.agents.trainer_template import build_trainer
from blinding_ray.agents.random.random_policy import MyRandomPolicy

RandomTrainer = build_trainer(
    name="Random",
    default_policy=MyRandomPolicy,
)
