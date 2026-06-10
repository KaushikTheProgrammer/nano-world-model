"""Objective functions for planning."""

import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(
    alpha: float = 1.0,
    base: float = 2.0,
    mode: str = "last",
    visual_metric: str = "mse",
    token_dim: int | None = None,
    eps: float = 1e-6,
    channels_first: bool = False,
    first_frame_index: int = 1,
):
    """
    Create objective function for planning.

    Args:
        alpha: Weight for proprioceptive loss
        base: Base for exponential weighting (only used for mode="all")
        mode: "last" (loss on final frame) or "all" (loss on all frames)
        visual_metric: "mse" for raw latent MSE, or "cosine" for token-wise
            cosine distance. The latter is useful for semantic patch features.
        token_dim: Token channel dimension for flattened semantic latents.
        channels_first: layout of the flattened visual dim. False (default) =
            token-major [..., n_tokens*token_dim] (tokens contiguous). True =
            channel-major [..., token_dim*n_tokens] — the layout produced by
            flattening a [C,h,w] latent (DiffusionWorldModel.rollout) — tokens
            are recovered as reshape(C, hw).T. Using the wrong layout silently
            mixes channels across tokens.
        first_frame_index: for mode="first": index of the first GENERATED frame
            in the rollout (context frames precede it; n_context=1 -> 1).
        mode: "last" (loss on final frame), "first" (loss on the first generated
            frame — the least autoregressively-degraded one, the chunk actually
            executed under execute-one-replan), or "all" (exponentially weighted).

    Returns:
        Objective function that takes (z_obs_pred, z_obs_tgt) and returns loss [B]
    """
    metric = nn.MSELoss(reduction="none")

    def visual_loss(pred_visual, tgt_visual):
        if visual_metric == "mse":
            return metric(pred_visual, tgt_visual).mean(dim=tuple(range(1, pred_visual.ndim)))

        if visual_metric == "cosine":
            pred_tokens = _to_tokens(pred_visual)
            tgt_tokens = _to_tokens(tgt_visual)
            pred_tokens = torch.nn.functional.normalize(pred_tokens, dim=-1, eps=eps)
            tgt_tokens = torch.nn.functional.normalize(tgt_tokens, dim=-1, eps=eps)
            loss = 1.0 - (pred_tokens * tgt_tokens).sum(dim=-1)
            return loss.mean(dim=tuple(range(1, loss.ndim)))

        raise NotImplementedError(f"Unknown visual_metric: {visual_metric}")

    def _to_tokens(x):
        if token_dim is None or token_dim <= 0:
            raise ValueError("objective.token_dim must be set for visual_metric='cosine'")
        if x.shape[-1] % token_dim != 0:
            raise ValueError(f"visual dim {x.shape[-1]} is not divisible by token_dim={token_dim}")
        if channels_first:                      # [..., C*hw] from [C,h,w] flatten -> [..., hw, C]
            return x.reshape(*x.shape[:-1], token_dim, -1).transpose(-1, -2)
        return x.reshape(*x.shape[:-1], -1, token_dim)

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on the last predicted frame.

        Args:
            z_obs_pred: dict
                - 'visual': [B, T, D_visual] predicted visual embeddings
                - 'proprio': [B, T, D_proprio] or None
            z_obs_tgt: dict
                - 'visual': [B, T, D_visual] target visual embeddings
                - 'proprio': [B, T, D_proprio] or None

        Returns:
            loss: [B] loss per batch element
        """
        # Visual loss
        loss_visual = visual_loss(
            z_obs_pred["visual"][:, -1:],
            z_obs_tgt["visual"][:, -1:],
        )

        # Proprioceptive loss (if available)
        if z_obs_pred.get("proprio") is not None and z_obs_tgt.get("proprio") is not None:
            loss_proprio = metric(
                z_obs_pred["proprio"][:, -1:],
                z_obs_tgt["proprio"][:, -1:]
            ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
            loss = loss_visual + alpha * loss_proprio
        else:
            loss = loss_visual

        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all predicted frames with exponential weighting.

        Args:
            z_obs_pred: dict
                - 'visual': [B, T, D_visual] predicted visual embeddings
                - 'proprio': [B, T, D_proprio] or None
            z_obs_tgt: dict
                - 'visual': [B, T, D_visual] target visual embeddings
                - 'proprio': [B, T, D_proprio] or None

        Returns:
            loss: [B] loss per batch element
        """
        T = z_obs_pred["visual"].shape[1]

        # Exponential weighting coefficients
        coeffs = np.array([base**i for i in range(T)], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        # Visual loss
        if visual_metric == "mse":
            loss_visual = metric(
                z_obs_pred["visual"],
                z_obs_tgt["visual"],
            ).mean(dim=tuple(range(2, z_obs_pred["visual"].ndim)))
        elif visual_metric == "cosine":
            pred_tokens = _to_tokens(z_obs_pred["visual"])
            tgt_tokens = _to_tokens(z_obs_tgt["visual"])
            pred_tokens = torch.nn.functional.normalize(pred_tokens, dim=-1, eps=eps)
            tgt_tokens = torch.nn.functional.normalize(tgt_tokens, dim=-1, eps=eps)
            loss_visual = 1.0 - (pred_tokens * tgt_tokens).sum(dim=-1)
            loss_visual = loss_visual.mean(dim=tuple(range(2, loss_visual.ndim)))
        else:
            raise NotImplementedError(f"Unknown visual_metric: {visual_metric}")
        loss_visual = (loss_visual * coeffs).mean(dim=1)

        # Proprioceptive loss (if available)
        if z_obs_pred.get("proprio") is not None and z_obs_tgt.get("proprio") is not None:
            loss_proprio = metric(
                z_obs_pred["proprio"],
                z_obs_tgt["proprio"]
            ).mean(dim=tuple(range(2, z_obs_pred["proprio"].ndim)))
            loss_proprio = (loss_proprio * coeffs).mean(dim=1)
            loss = loss_visual + alpha * loss_proprio
        else:
            loss = loss_visual

        return loss

    def objective_fn_first(z_obs_pred, z_obs_tgt):
        """Loss on the FIRST generated frame (+1 chunk): least WM-degraded, and the one
        actually executed under execute-one-replan. Goal target = its last frame."""
        i = first_frame_index
        loss_visual = visual_loss(
            z_obs_pred["visual"][:, i:i + 1],
            z_obs_tgt["visual"][:, -1:],
        )
        if z_obs_pred.get("proprio") is not None and z_obs_tgt.get("proprio") is not None:
            loss_proprio = metric(
                z_obs_pred["proprio"][:, i:i + 1],
                z_obs_tgt["proprio"][:, -1:]
            ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
            loss_visual = loss_visual + alpha * loss_proprio
        return loss_visual

    if mode == "last":
        return objective_fn_last
    elif mode == "first":
        return objective_fn_first
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
