"""
GT-action rollout visualization (NanoNAV).

Rolls the trained world model forward from ONE context frame under the dataset's ground-truth
(Δx, Δθ) actions and decodes the predicted latents back to RGB, so you can *see* the predicted
future vs the real future. Reuses the same machinery as action_diagnostic.py (load_checkpoint,
DiffusionWorldModel.rollout) — it does NOT need sample_dfot's flattened sampling config.

Per sample it writes:
  * sample<i>_grid.png  — rows [GT | Pred], cols [context, t+1 … t+H]
  * sample<i>_cmp.mp4   — side-by-side GT | Pred over time

The "GT" row is the VAE round-trip of the real frames (encode→decode), so GT and Pred share the
exact same decode path and pixel range — this isolates the world model's prediction error from VAE
reconstruction error.

    python src/sample/gt_rollout_viz.py --ckpt <run>/checkpoints/.../X.ckpt --out <run>/gt_rollout
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])  # put src/ on the path

from action_diagnostic import load_checkpoint           # noqa: E402  (reuse checkpoint loading)
from planning import DiffusionWorldModel                 # noqa: E402
from wm_datasets import create_train_val_datasets        # noqa: E402
from sampling_utils import decode_latents, save_comparison_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="gt_rollout")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, latent_codec, diffusion, train_cfg = load_checkpoint(args.ckpt, device)
    wm_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    world_model = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)
    vae = latent_codec.vae
    vae_precision = getattr(latent_codec, "precision", "fp32")

    H = train_cfg.model.num_frames - train_cfg.model.n_context_frames
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=train_cfg.dataset.frame_interval,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    loader = torch.utils.data.DataLoader(val, batch_size=args.num_samples, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    video = batch["video"].to(device)        # [B, 1+H, C, H, W]
    action = batch["action"].to(device)      # [B, 1+H, 2]  normalized (Δx, Δθ)
    B = video.shape[0]

    obs_0 = {"visual": video[:, :1]}          # one context frame
    act_gt = action[:, :H]                     # [B, H, 2]

    print(f"Rolling out {B} samples under GT actions (H={H}, {args.num_sampling_steps} DDIM steps)...")
    z_obses, _ = world_model.rollout(obs_0, act_gt, num_sampling_steps=args.num_sampling_steps)
    pred_flat = z_obses["visual"]              # [B, 1+H, D]

    # Latent grid shape from a reference encode, to un-flatten the rollout latents.
    ref = latent_codec.encode(video[:, 0])     # [B, C_lat, h, w]
    C_lat, hl, wl = ref.shape[1:]
    pred_lat = pred_flat.reshape(B, 1 + H, C_lat, hl, wl)
    gt_lat = torch.stack([latent_codec.encode(video[:, t]) for t in range(1 + H)], dim=1)

    pred_frames = decode_latents(vae, pred_lat, vae_precision).clamp(0, 1).cpu().numpy()  # [B,1+H,C,H,W]
    gt_frames = decode_latents(vae, gt_lat, vae_precision).clamp(0, 1).cpu().numpy()

    # Per-frame latent-L2 (pred vs gt) for annotation.
    l2 = torch.norm((pred_lat - gt_lat).reshape(B, 1 + H, -1), dim=-1).cpu().numpy()

    titles = ["context"] + [f"t+{i}" for i in range(1, 1 + H)]
    for i in range(B):
        fig, axes = plt.subplots(2, 1 + H, figsize=(3 * (1 + H), 6))
        for t in range(1 + H):
            axes[0, t].imshow(np.transpose(gt_frames[i, t], (1, 2, 0)))
            axes[0, t].set_title(f"GT {titles[t]}")
            axes[0, t].axis("off")
            axes[1, t].imshow(np.transpose(pred_frames[i, t], (1, 2, 0)))
            axes[1, t].set_title(f"Pred {titles[t]}" + ("" if t == 0 else f"\nlatentL2={l2[i, t]:.1f}"))
            axes[1, t].axis("off")
        fig.suptitle(f"GT-action rollout — sample {i} (top: real via VAE, bottom: world-model prediction)")
        fig.tight_layout()
        fig.savefig(out / f"sample{i}_grid.png", dpi=110)
        plt.close(fig)
        save_comparison_video(
            torch.from_numpy(gt_frames[i]), torch.from_numpy(pred_frames[i]),
            str(out / f"sample{i}_cmp.mp4"), fps=args.fps,
        )

    mean_l2 = float(l2[:, 1:].mean())
    print(f"Saved {B} grids + comparison videos -> {out}")
    print(f"Mean predicted-vs-GT latent L2 over horizon: {mean_l2:.3f}")


if __name__ == "__main__":
    main()
