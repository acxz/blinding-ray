# Based off of MARWIL impl: https://github.com/ray-project/ray/blob/master/rllib/agents/marwil/marwil_torch_policy.py

from typing import Dict, Optional

import gym

from ray.rllib.agents.a3c.a3c_torch_policy import ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance
from ray.rllib.utils.typing import PolicyID, TensorType, TrainerConfigDict

import blinding_ray


torch, _ = try_import_torch()


def state_visitation_frequencies():
    """Calculate state visitation frequency Ds for each state s under a
    given policy pi.

    You can get pi from compute_policy.
    """
    pass


def mceirl_loss(
    self,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
) -> TensorType:
    """Constructs the loss for Maximum Casual Entropy.

    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The MCEIRL loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    observations = train_batch[SampleBatch.OBS]
    actions = train_batch[SampleBatch.ACTIONS]
    total_loss = 0

    for observation in observations:
        obs = 0

    average_feature_under_input_behavior = 0
    expected_feature_under_learned_policy = 0

    import pdb
    pdb.set_trace()

    # return super().loss(model, dist_class, train_batch)


def marwil_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
) -> TensorType:
    model_out, _ = model(train_batch)
    action_dist = dist_class(model_out, model)
    actions = train_batch[SampleBatch.ACTIONS]
    # log\pi_\theta(a|s)
    logprobs = action_dist.logp(actions)

    # Advantage estimation.
    if policy.config["beta"] != 0.0:
        cumulative_rewards = train_batch[Postprocessing.ADVANTAGES]
        state_values = model.value_function()
        adv = cumulative_rewards - state_values
        adv_squared_mean = torch.mean(torch.pow(adv, 2.0))

        explained_var = explained_variance(cumulative_rewards, state_values)
        policy.explained_variance = torch.mean(explained_var)

        # Policy loss.
        # Update averaged advantage norm.
        rate = policy.config["moving_average_sqd_adv_norm_update_rate"]
        policy._moving_average_sqd_adv_norm.add_(
            rate * (adv_squared_mean - policy._moving_average_sqd_adv_norm)
        )
        # Exponentially weighted advantages.
        exp_advs = torch.exp(
            policy.config["beta"]
            * (adv / (1e-8 + torch.pow(policy._moving_average_sqd_adv_norm, 0.5)))
        ).detach()
        # Value loss.
        policy.v_loss = 0.5 * adv_squared_mean
    else:
        # Policy loss (simple BC loss term).
        exp_advs = 1.0
        # Value loss.
        policy.v_loss = 0.0

    # logprob loss alone tends to push action distributions to
    # have very low entropy, resulting in worse performance for
    # unfamiliar situations.
    # A scaled logstd loss term encourages stochasticity, thus
    # alleviate the problem to some extent.
    logstd_coeff = policy.config["bc_logstd_coeff"]
    if logstd_coeff > 0.0:
        logstds = torch.mean(action_dist.log_std, dim=1)
    else:
        logstds = 0.0

    policy.p_loss = -torch.mean(exp_advs * (logprobs + logstd_coeff * logstds))

    # Combine both losses.
    policy.total_loss = policy.p_loss + \
        policy.config["vf_coeff"] * policy.v_loss

    return policy.total_loss


def postprocess_advantages(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[PolicyID, SampleBatch]] = None,
    episode=None,
) -> SampleBatch:
    """Postprocesses a trajectory and returns the processed trajectory.
    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.
    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[Episode]): Optional multi-agent episode
            object in which the agents operated.
    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        index = "last" if SampleBatch.NEXT_OBS in sample_batch else -1
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index=index
        )
        last_r = policy._value(**input_dict)

    # Adds the "advantages" (which in the case of MARWIL are simply the
    # discounted cummulative rewards) to the SampleBatch.
    return compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        # We just want the discounted cummulative rewards, so we won't need
        # GAE nor critic (use_critic=True: Subtract vf-estimates from returns).
        use_gae=False,
        use_critic=False,
    )


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    stats = {
        "policy_loss": policy.p_loss,
        "total_loss": policy.total_loss,
    }
    if policy.config["beta"] != 0.0:
        stats["moving_average_sqd_adv_norm"] = policy._moving_average_sqd_adv_norm
        stats["vf_explained_var"] = policy.explained_variance
        stats["vf_loss"] = policy.v_loss

    return stats


def setup_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    # Setup Value branch of our NN.
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)

    # Not needed for pure BC.
    if policy.config["beta"] != 0.0:
        # Set up a torch-var for the squared moving avg. advantage norm.
        policy._moving_average_sqd_adv_norm = torch.tensor(
            [policy.config["moving_average_sqd_adv_norm_start"]],
            dtype=torch.float32,
            requires_grad=False,
        ).to(policy.device)


MCEIRLTorchPolicy = build_policy_class(
    name="MCEIRLTorchPolicy",
    framework="torch",
    loss_fn=marwil_loss,
    get_default_config=lambda: blinding_ray.agents.mceirl.mceirl.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_advantages,
    extra_grad_process_fn=apply_grad_clipping,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin],
)
