# Copyright (c) 2026, PRIN Lab, NII
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules import HiddenState


class L2C2:
    """Smoothing regularization in local space.

    References:
        - Kobayashi "L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth Reinforcement Learning." IROS (2022).
        - Kobayashi & Yamanaka "Redesigning Regularization for Effective Policy Smoothing." RA-L (2026).
    """

    def __init__(
        self,
        obs_dims: tuple[int, int],
        lambdas: tuple[float, float] = (1e-5, 1e-3),
        scale_local_space: float = 1.0,
        gain_value_func: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        """Initialize the l2c2 extension.

        Args:
            obs_dims: observation dimensions for actor and critic, which are utilized for computing power coeff.
            lambdas: lower and upper bounds of weight for regularization.
            scale_local_space: scale of local space, which is defined by state transition.
            gain_value_func: weight scale for smoothing value function.
            eps: small value for numerical stabilization of normalization in logsumexp (just in case).
        """
        # l2c2 parameters
        self.eps = lambdas[0] / (lambdas[1] - lambdas[0])
        self.lambd = lambdas[1] * self.eps
        m_ = math.exp(0.5 * (math.log(lambdas[0]) + math.log(lambdas[1])))
        self.power_actor = obs_dims[0] * (1.0 - lambdas[1] * self.eps / m_)
        self.power_critic = obs_dims[1] * (1.0 - lambdas[1] * self.eps / m_)
        self.scale_local_space = scale_local_space
        self.gain_value_func = gain_value_func
        self.eps_logsumexp = eps

    def _logsumexp(self, losses):
        scale = losses.data.mean() + self.eps_logsumexp
        return scale * torch.logsumexp(losses / scale, 0)


    def compute_loss(
        self,
        observations: TensorDict, next_observations: TensorDict, dones: torch.Tensor,
        policies: Distribution, actor: MLPModel,
        values: torch.Tensor | None = None, critic: MLPModel | None = None,
        masks: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None),
        ) -> torch.Tensor:
        """Compute the l2c2 loss.

        If values and critic are also given and gain_value_func is positive, l2c2 makes the value function smooth as well.
        Assume that hidden states are shared in local space.
        """
        m_ = ~dones.bool().flatten()
        noise = torch.zeros_like(observations).apply_(lambda x_: x_.uniform_(-1.0, 1.0))
        norm = torch.cat([v_.abs() for v_ in noise.to_dict().values()], dim=-1).amax(dim=-1)
        local_observations = observations + self.scale_local_space * (next_observations - observations) * noise
        #
        actor(local_observations, masks=masks, hidden_state=hidden_states[0], stochastic_output=True)
        local_policies = actor.distribution._distribution
        a_ = local_policies.rsample()
        logq = policies.log_prob(a_).sum(dim=-1)
        logp = local_policies.log_prob(a_).sum(dim=-1)
        #
        gain_actor = self.lambd / (norm ** self.power_actor + self.eps)
        loss_actor = gain_actor * (- logq + logp + (logq - logp).exp() - 1.0)
        loss_actor = loss_actor[m_]
        loss = self._logsumexp(loss_actor)
        #
        if values is not None and critic is not None and self.gain_value_func > 0.0:
            local_values = critic(local_observations, masks=masks, hidden_state=hidden_states[1])
            gain_critic = self.lambd / (norm ** self.power_critic + self.eps)
            loss_critic = gain_critic * (values - local_values).square().flatten()
            loss_critic = loss_critic[m_]
            loss = loss + self.gain_value_func * self._logsumexp(loss_critic)
        #
        return loss


def resolve_l2c2_config(alg_cfg: dict, obs_dims: tuple[int, int]) -> dict:
    """Resolve the l2c2 configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs_dims: Observation dimensions for actor and critic

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using l2c2 then pass the obs_dims
    if "l2c2_cfg" in alg_cfg and alg_cfg["l2c2_cfg"] is not None:
        alg_cfg["l2c2_cfg"]["obs_dims"] = obs_dims
    else:
        alg_cfg["l2c2_cfg"] = None
    return alg_cfg
