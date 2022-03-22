from ray.rllib.agents.trainer_template import build_trainer
from blinding_ray.agents.attacker.attacker_policy import AttackerPolicy

AttackerTrainer = build_trainer(
    name="Attacker",
    default_policy=AttackerPolicy,
)
