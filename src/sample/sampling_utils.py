"""
Shared utility functions for video sampling and rollout scripts.
Provides VAE encoding/decoding, video saving, and frame resizing.
"""

import torch
import imageio
from einops import rearrange
from omegaconf import OmegaConf

from utils.vae_ops import encode_first_stage, decode_first_stage

# Saved training configs carry ${hydra:...} interpolations (e.g. planning.output_dir) that only
# resolve inside a live Hydra run. The sampling/rollout scripts load + resolve those configs
# standalone, so register a stub here (imported by sample_dfot.py and rollout.py) to avoid a crash.
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda *a, **k: ".")


def encode_frames(vae, frames, device, vae_precision: str = "fp32"):
    """Encode [B,F,C,H,W] frames to [B,F,C_lat,H/8,W/8] scaled latents."""
    B, F, C, H, W = frames.shape
    frames = frames.to(device)
    frames_flat = rearrange(frames, 'b f c h w -> (b f) c h w')
    with torch.no_grad():
        latents = encode_first_stage(vae, frames_flat, precision=vae_precision)
    return rearrange(latents, '(b f) c h w -> b f c h w', b=B)


def decode_latents(vae, latents, vae_precision: str = "fp32"):
    """Decode [B,F,C_lat,H/8,W/8] scaled latents to [B,F,C,H,W] frames in [0,1]."""
    B, F = latents.shape[:2]
    latents_flat = rearrange(latents, 'b f c h w -> (b f) c h w')
    with torch.no_grad():
        frames = decode_first_stage(vae, latents_flat, precision=vae_precision)
    frames = rearrange(frames, '(b f) c h w -> b f c h w', b=B)
    return ((frames + 1) / 2).clamp(0, 1)


def save_video(frames, save_path, fps):
    """
    Save frames as an MP4 video file.

    Args:
        frames: [F, C, H, W] tensor in range [0, 1]
        save_path: Output file path
        fps: Frames per second
    """
    video = (frames * 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    imageio.mimwrite(save_path, video, fps=fps, quality=9)


def save_comparison_video(gt_frames, pred_frames, save_path, fps):
    """
    Save side-by-side comparison video (ground truth | prediction).

    Args:
        gt_frames: [F, C, H, W] tensor in range [0, 1]
        pred_frames: [F, C, H, W] tensor in range [0, 1]
        save_path: Output file path
        fps: Frames per second
    """
    comparison = torch.cat([gt_frames, pred_frames], dim=3)
    save_video(comparison, save_path, fps)


def resize_frames(frames, image_size, resize_mode):
    """
    Resize video frames to target size.

    Replicates the stretch/pad resize logic from WorldModelDataset._load_slice().

    Args:
        frames: [T, C, H, W] tensor
        image_size: (H, W) tuple
        resize_mode: "stretch" or "pad"

    Returns:
        Resized [T, C, H, W] tensor
    """
    if frames.shape[-2:] == tuple(image_size):
        return frames

    if resize_mode == "stretch":
        return torch.nn.functional.interpolate(
            frames, size=image_size, mode='bilinear', align_corners=False
        )
    elif resize_mode == "pad":
        T, C, H, W = frames.shape
        target_h, target_w = image_size
        scale = min(target_h / H, target_w / W)
        new_h = int(H * scale)
        new_w = int(W * scale)
        frames = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        frames = torch.nn.functional.pad(
            frames,
            (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
            value=0.0
        )
        return frames
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")
