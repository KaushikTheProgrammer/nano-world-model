"""CEM (Cross-Entropy Method) Planner for world model planning."""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class CEMPlanner:
    """Cross-Entropy Method planner for trajectory optimization."""

    def __init__(
        self,
        world_model,
        objective_fn,
        action_dim: int,
        horizon: int,
        num_samples: int = 100,
        topk: int = 10,
        opt_steps: int = 5,
        var_scale: float = 1.0,
        eval_every: int = 1,
        name: str = "CEM",
        device: str = "cuda",
        sigma_min: float = 1e-3,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        rollout_batch_size: Optional[int] = None,
    ):
        """
        Initialize CEM planner.

        Args:
            world_model: World model for rollout (must have encode_obs and rollout methods)
            objective_fn: Objective function to minimize
            action_dim: Dimension of action space
            horizon: Planning horizon (number of timesteps)
            num_samples: Number of action sequences to sample per iteration
            topk: Number of best samples to use for updating distribution
            opt_steps: Number of optimization iterations
            var_scale: Initial variance scale for action sampling
            eval_every: Evaluation frequency
            name: Name for logging
            device: Device to run on
        """
        self.world_model = world_model
        self.objective_fn = objective_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.topk = topk
        self.opt_steps = opt_steps
        self.var_scale = var_scale
        self.eval_every = eval_every
        self.name = name
        self.device = device
        # Prevents late-iteration variance collapse (all top-k identical).
        self.sigma_min = sigma_min
        # Optional clip range to keep samples in-distribution.
        self.action_low = action_low
        self.action_high = action_high
        self.rollout_batch_size = rollout_batch_size

    def _compute_losses_for_samples(
        self,
        obs_0_single: Dict[str, torch.Tensor],
        z_obs_g_single: Dict[str, Optional[torch.Tensor]],
        action_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out CEM samples, optionally chunked to avoid large SDPA kernels."""
        if self.rollout_batch_size is None or self.rollout_batch_size <= 0:
            rollout_batch_size = self.num_samples
        else:
            rollout_batch_size = min(int(self.rollout_batch_size), self.num_samples)

        losses = []
        for start in range(0, self.num_samples, rollout_batch_size):
            end = min(start + rollout_batch_size, self.num_samples)
            cur_actions = action_samples[start:end]
            cur_n = cur_actions.shape[0]

            obs_0_expanded = {
                k: v.expand(cur_n, *v.shape[1:])
                for k, v in obs_0_single.items()
            }
            z_obs_g_expanded = {
                k: v.expand(cur_n, *v.shape[1:]) if v is not None else None
                for k, v in z_obs_g_single.items()
            }

            with torch.no_grad():
                z_obses, _ = self.world_model.rollout(
                    obs_0=obs_0_expanded,
                    act=cur_actions,
                )
                losses.append(self.objective_fn(z_obses, z_obs_g_expanded))

        return torch.cat(losses, dim=0)

    def init_mu_sigma(self, batch_size: int, actions: Optional[torch.Tensor] = None):
        """
        Initialize mean and variance for action distribution.

        Args:
            batch_size: Batch size
            actions: Optional initial actions [B, T, action_dim] (T <= horizon)

        Returns:
            mu: Mean [B, horizon, action_dim]
            sigma: Std [B, horizon, action_dim]
        """
        sigma = self.var_scale * torch.ones([batch_size, self.horizon, self.action_dim])

        if actions is None:
            mu = torch.zeros(batch_size, self.horizon, self.action_dim)
        else:
            mu = actions.clone()
            t = mu.shape[1]
            remaining_t = self.horizon - t

            if remaining_t > 0:
                new_mu = torch.zeros(batch_size, remaining_t, self.action_dim)
                mu = torch.cat([mu, new_mu.to(mu.device)], dim=1)

        return mu, sigma

    def plan(
        self,
        obs_0: Dict[str, torch.Tensor],
        obs_g: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        return_elites: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Plan action sequence to reach goal observation.

        Args:
            obs_0: Initial observation dict
                - 'visual': [B, 1, C, H, W] initial frame
            obs_g: Goal observation dict
                - 'visual': [B, 1, C, H, W] goal frame
            actions: Optional initial action sequence [B, T, action_dim]

        Returns:
            actions: Optimized action sequence [B, horizon, action_dim]
            info: Planning information dict
        """
        batch_size = obs_0["visual"].shape[0]

        # Move both observations to device BEFORE any encode call so we don't
        # mix devices in the objective.
        obs_0 = {k: v.to(self.device) for k, v in obs_0.items()}
        obs_g = {k: (v.to(self.device) if v is not None else None) for k, v in obs_g.items()}

        # Encode goal observation
        with torch.no_grad():
            z_obs_g = self.world_model.encode_obs(obs_g)

        # Initialize action distribution
        mu, sigma = self.init_mu_sigma(batch_size, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        losses_history = []
        elite_actions = None  # final-iteration top-k elites (for viz), if return_elites

        for i in range(self.opt_steps):
            # Optimize each instance in batch
            batch_losses = []

            for b in range(batch_size):
                # Sample action sequences
                action_samples = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                    * sigma[b]
                    + mu[b]
                )
                action_samples[0] = mu[b]  # First sample is current mean
                # Clip to action space if bounds provided.
                if self.action_low is not None or self.action_high is not None:
                    action_samples = action_samples.clamp(
                        min=self.action_low if self.action_low is not None else -float("inf"),
                        max=self.action_high if self.action_high is not None else float("inf"),
                    )

                obs_0_single = {k: v[b:b+1] for k, v in obs_0.items()}
                z_obs_g_single = {
                    k: v[b:b+1] if v is not None else None
                    for k, v in z_obs_g.items()
                }
                loss = self._compute_losses_for_samples(
                    obs_0_single=obs_0_single,
                    z_obs_g_single=z_obs_g_single,
                    action_samples=action_samples,
                )

                # Select top-k
                topk_idx = torch.argsort(loss)[:self.topk]
                topk_actions = action_samples[topk_idx]
                batch_losses.append(loss[topk_idx[0]].item())

                # Update distribution, flooring sigma so it doesn't collapse.
                mu[b] = topk_actions.mean(dim=0)
                sigma[b] = topk_actions.std(dim=0).clamp(min=self.sigma_min)

                # Surface the final iteration's elites for visualization (the candidates CEM
                # selected). batch_size is 1 in the LeKiwi path, so the last b wins.
                if return_elites and i == self.opt_steps - 1:
                    elite_actions = topk_actions.detach().cpu()  # [topk, horizon, action_dim], normalized

            avg_loss = np.mean(batch_losses)
            losses_history.append(avg_loss)

            if (i + 1) % self.eval_every == 0:
                print(f"  {self.name} iteration {i+1}/{self.opt_steps}: loss = {avg_loss:.4f}")

        info = {
            "losses": losses_history,
            "final_loss": losses_history[-1],
            "num_iterations": self.opt_steps,
        }
        if return_elites:
            info["elite_actions"] = elite_actions  # [topk, horizon, action_dim] (cpu) or None

        return mu, info
