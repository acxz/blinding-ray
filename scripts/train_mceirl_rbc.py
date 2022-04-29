from datetime import datetime
import os
import tempfile

import pyspiel

from ray.rllib.agents.callbacks import MultiCallbacks
# from ray.rllib.agents.trainer import Trainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.tune.logger import pretty_print
from ray.tune.logger import UnifiedLogger

from blinding_ray.agents.attacker import AttackerCallbacks, AttackerPolicy
from blinding_ray.agents.mceirl import MCEIRLTorchPolicy, MCEIRLTrainer
from blinding_ray.agents.random import RandomPolicy
from blinding_ray.agents.trout import TroutCallbacks, TroutPolicy
from blinding_ray.env.open_spiel_rbc import OpenSpielRbcEnv

# RBC OpenSpiel env
register_env("open_spiel_rbc_env", lambda _: OpenSpielRbcEnv(
    pyspiel.load_game("rbc")))


def policy_mapping_fn(agent_id, episode, worker,
                      player="mceirl",
                      opponent="attacker",
                      **kwargs):
    # TODO: Playing trout vs trout may not work as the policiesID will overlap
    # Need to use policy state instead of just the self to prevent this issue
    # Same issue with attacker vs attacker (both attackers will end up using the
    # same strategy)
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

    # Training wheels:
    # For making the simplest dataset, ensure that trout is always white
    # Ensure attacker is always black, ensure attacker only uses one strat
    # See attacker_callbacks for this
    if agent_id == 1:
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
    "record_env": True,
    "log_sys_usage": True,
    "evaluation_num_workers": 1,
    "evaluation_duration": 1,  # Play for one episode
    "evaluation_config": {
        "render_env": False,  # This does not work if record_env is False
        # Is this even recording? (I want it to record the ascii render as well"
        "record_env": True,
    },
    "num_gpus": 0,
    "output": "logdir",
    # "output": "dataset", # not working at all for me
    # "output_config": {
    #   "format": "json",
    #   "path": "/home/acxz/vcs/git/github/acxz/blinding-ray/logs/datasets/AttackerRandom/",
    # },
    "output_compress_columns": ["obs", "new_obs"],
    "multiagent": {
        "policies": {
            "attacker": PolicySpec(policy_class=AttackerPolicy),
            "random": PolicySpec(policy_class=RandomPolicy),
            "trout": PolicySpec(policy_class=TroutPolicy),
            "mceirl": PolicySpec(policy_class=MCEIRLTorchPolicy),
        },
        "policy_mapping_fn": policy_mapping_fn,
        # "policies_to_train": [MCEIRLTorchPolicy], # this doesn't train/call loss func
    },
    "callbacks": MultiCallbacks([
        AttackerCallbacks,
        #TroutCallbacks,
    ]),
    "framework": "torch",
    # behavior data config options
    # for training from dataset
    "input": "~/vcs/git/github/acxz/blinding-ray/logs/BCAttackerTrout1000_2022-04-27_22-25-43mi4bm491/output-2022-04-27_22-25-58_worker-1_0.json",
    # "input": "/tmp/attackervrandom/",
    # input cannot read json for multiagent batch, but output can't output in
    # dataset api, hmmm.... dilemma
    # "input": "sampler", # For eval
    # Alternative is to extend config from mceirl via
    # mceirl.DEFAULT_CONFIG.copy()
    # Use importance sampling estimators for reward.
    # "input_evaluation": ["is", "wis"],
    "input_evaluation": [],  # Off-policy estimation is not implemented for
    # multiagent (error doesn't show up if using input: sampler)
    # === Postprocessing/accum., discounted return calculation ===
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf in
    # case an input line ends with a non-terminal timestep.
    "use_gae": True,
    # Whether to calculate cumulative rewards. Must be True.
    "postprocess_inputs": False,
    # === Training ===
    # Scaling of advantages in exponential terms.
    # When beta is 0.0, MARWIL is reduced to behavior cloning
    # (imitation learning); see bc.py algorithm in this same directory.
    "beta": 0.0,
    # Balancing value estimation loss and policy optimization loss.
    "vf_coeff": 1.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Learning rate for Adam optimizer.
    "lr": 1e-4,
    # The squared moving avg. advantage norm (c^2) update rate
    # (1e-8 in the paper).
    "moving_average_sqd_adv_norm_update_rate": 1e-8,
    # Starting value for the squared moving avg. advantage norm (c^2).
    "moving_average_sqd_adv_norm_start": 100.0,
    # Number of (independent) timesteps pushed through the loss
    # each SGD round.
    "train_batch_size": 2000,
    # Size of the replay buffer in (single and independent) timesteps.
    # The buffer gets filled by reading from the input files line-by-line
    # and adding all timesteps on one line at once. We then sample
    # uniformly from the buffer (`train_batch_size` samples) for
    # each training step.
    "replay_buffer_size": 10000,
    # Number of steps to read before learning starts.
    "learning_starts": 0,
    # A coeff to encourage higher action distribution entropy for exploration.
    "bc_logstd_coeff": 0.0,
}


def custom_log_creator(custom_path, custom_str):
    # Create custom logger to save results in specified path
    # See: https://stackoverflow.com/a/66160412/18508790

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(_config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(_config, logdir, loggers=None)

    return logger_creator


# Create our RLlib trainer
RESULTS_DIR = "~/vcs/git/github/acxz/blinding-ray/logs"
trainer = MCEIRLTrainer(
    config=config,
    logger_creator=custom_log_creator(os.path.expanduser(RESULTS_DIR),
                                      # 'AttackerTrout100')
                                      'BCTrainAttackerTrout1000')
)

# Operational Checklist
# Gen behavior data by setting policy_mapping_fn to attacker v. trout, custom dir to
# dataset name, clear input of dataset folder, change output dir, using eval
# loop, change input to sampler
# Train MCEIRL by setting policy_mapping_fn to mceirl v. random (maybe attacker
# since it has seen experiences with it), (make sure to comment TroutCallbacks)
# custom dir to mceirl name, set input to dataset folder, change output dir, using training loop
# Evaluate MCEIRL with the same as above but, load in trained checkpoint and use
# eval loop and comment out train loop, make sure to change input to sampler (or
# comment out existing input)
# may want to set render to true to see in print outs as well as policy id
# mapping

# Run training iterations
print("Training")
TRAIN_ITERATIONS = 100000
for train_iter in range(TRAIN_ITERATIONS):
    print("Train iter: " + str(train_iter + 1) + "/" + str(TRAIN_ITERATIONS))
    result = trainer.train()
    # print(pretty_print(result))
    if (train_iter + 1) % 1000 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

# make sure to change the file and the directory for checkpoint
# Run evaluation
#checkpoint_path = "/home/acxz/vcs/git/github/acxz/blinding-ray/logs/BCTrainAttackerTrout1000_2022-04-28_00-25-58d6fdk34l/checkpoint_061000/checkpoint-61000"
# trainer.restore(checkpoint_path)
# print("Evaluating")
#EVAL_ITERATIONS = 1000
# for eval_iter in range(EVAL_ITERATIONS):
#    print("Eval iter: " + str(eval_iter + 1) + "/" + str(EVAL_ITERATIONS))
#    trainer.evaluate()
