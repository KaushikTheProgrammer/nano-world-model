"""
Per-chunk motion vs frame-difference diagnostic (NanoNAV).

Probes *why* action-conditioning is weak (action_diagnostic FAIL): how much does the robot actually
move per chunk, and how much does the image the model sees actually change over that same chunk?

For consecutive chunk-endpoint frames (frame k and frame k+f, i.e. one frame_interval / 167 ms apart
— exactly the pairs the world model is trained on), it measures:
  * physical motion   : un-normalized (Δx [cm], Δθ [deg]) for the chunk
  * pixel difference  : mean |Δpixel| in % (the literal change in the frames fed to the model)
  * SD-VAE latent L2  : ||z(k+f) - z(k)|| — the signal the action branch must learn from
                        (training.md: if this is barely above noise, there is little to learn).

Outputs (to --out):
  * chunk_examples.png   — example pairs [frame k | frame k+f | |diff| heatmap], annotated
  * chunk_distributions.png — histograms of |Δx|, pixel-change %, latent L2, and a motion-vs-latentL2 scatter
  * chunk_motion_stats.json — summary numbers

    python src/sample/chunk_motion_viz.py --ckpt <run>/checkpoints/.../X.ckpt --out <run>/chunk_motion
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])

from action_diagnostic import load_checkpoint           # noqa: E402  (reuse ckpt + VAE loading)
from wm_datasets import create_train_val_datasets        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="any checkpoint from the run (used only for its SD-VAE + config)")
    ap.add_argument("--out", default="chunk_motion")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-examples", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, latent_codec, _, train_cfg = load_checkpoint(args.ckpt, device)

    # un-normalization stats for the integrated action (set when the dataset is built)
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=train_cfg.dataset.frame_interval,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    a_mean = np.asarray(val._raw_action_mean).reshape(-1)[:2]   # (Δx [m], Δθ [rad])
    a_std = np.asarray(val._raw_action_std).reshape(-1)[:2]
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dx_cm, dth_deg, pix_pct, lat_l2 = [], [], [], []
    examples = []   # (frame_k, frame_kf, dx_cm, dth_deg, pix, l2)

    n = 0
    for batch in loader:
        if n >= args.n_batches:
            break
        video = batch["video"].to(device)        # [B, 1+H, C, H, W], pixels in [-1,1]
        action = batch["action"].cpu().numpy()    # [B, 1+H, 2] normalized
        B, T = video.shape[:2]
        with torch.no_grad():
            for t in range(T - 1):                # each consecutive pair = one chunk
                z0 = latent_codec.encode(video[:, t])
                z1 = latent_codec.encode(video[:, t + 1])
                l2 = torch.norm((z1 - z0).reshape(B, -1), dim=-1).cpu().numpy()
                f0 = ((video[:, t].clamp(-1, 1) + 1) / 2)
                f1 = ((video[:, t + 1].clamp(-1, 1) + 1) / 2)
                pix = (f1 - f0).abs().mean(dim=(1, 2, 3)).cpu().numpy() * 100.0  # % of full range
                act_phys = action[:, t] * a_std + a_mean      # [B,2] -> (Δx m, Δθ rad)
                dx_cm.extend((act_phys[:, 0] * 100.0).tolist())
                dth_deg.extend((np.degrees(act_phys[:, 1])).tolist())
                pix_pct.extend(pix.tolist()); lat_l2.extend(l2.tolist())
                if len(examples) < args.n_examples and t == 0:
                    for b in range(min(B, args.n_examples - len(examples))):
                        examples.append((f0[b].cpu().numpy(), f1[b].cpu().numpy(),
                                         act_phys[b, 0] * 100, np.degrees(act_phys[b, 1]), pix[b], l2[b]))
        n += 1

    dx_cm, dth_deg, pix_pct, lat_l2 = map(np.asarray, (dx_cm, dth_deg, pix_pct, lat_l2))

    # ---- examples montage ----
    ne = len(examples)
    fig, axes = plt.subplots(ne, 3, figsize=(10, 3 * ne))
    if ne == 1:
        axes = axes[None, :]
    for i, (f0, f1, dx, dth, pix, l2) in enumerate(examples):
        diff = np.abs(f1 - f0).mean(0)
        axes[i, 0].imshow(np.transpose(f0, (1, 2, 0))); axes[i, 0].set_title("frame k"); axes[i, 0].axis("off")
        axes[i, 1].imshow(np.transpose(f1, (1, 2, 0)))
        axes[i, 1].set_title(f"frame k+f  (Δx={dx:.2f}cm, Δθ={dth:.1f}°)"); axes[i, 1].axis("off")
        im = axes[i, 2].imshow(diff, cmap="inferno", vmin=0, vmax=max(diff.max(), 1e-3))
        axes[i, 2].set_title(f"|diff|  pix={pix:.2f}%  latL2={l2:.1f}"); axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046)
    fig.suptitle("Per-chunk (167 ms) frame difference — the change the world model is asked to predict")
    fig.tight_layout(); fig.savefig(out / "chunk_examples.png", dpi=110); plt.close(fig)

    # ---- distributions ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].hist(np.abs(dx_cm), bins=60, color="steelblue"); ax[0, 0].set_title("|Δx| per chunk (cm)"); ax[0, 0].set_xlabel("cm")
    ax[0, 1].hist(pix_pct, bins=60, color="seagreen"); ax[0, 1].set_title("mean |Δpixel| per chunk (% of range)"); ax[0, 1].set_xlabel("%")
    ax[1, 0].hist(lat_l2, bins=60, color="darkred"); ax[1, 0].set_title("SD-VAE latent L2 per chunk (the action signal)"); ax[1, 0].set_xlabel("L2")
    ax[1, 1].scatter(np.abs(dx_cm), lat_l2, s=4, alpha=0.3); ax[1, 1].set_title("motion vs latent change"); ax[1, 1].set_xlabel("|Δx| (cm)"); ax[1, 1].set_ylabel("latent L2")
    fig.suptitle("Per-chunk motion vs frame/latent change — weak motion ⇒ weak signal for the action branch")
    fig.tight_layout(); fig.savefig(out / "chunk_distributions.png", dpi=110); plt.close(fig)

    corr = float(np.corrcoef(np.abs(dx_cm), lat_l2)[0, 1])
    stats = {
        "n_chunks": int(len(dx_cm)),
        "dx_cm": {"mean": float(np.abs(dx_cm).mean()), "p50": float(np.median(np.abs(dx_cm))), "p95": float(np.percentile(np.abs(dx_cm), 95)), "max": float(np.abs(dx_cm).max())},
        "dtheta_deg": {"mean_abs": float(np.abs(dth_deg).mean()), "p95": float(np.percentile(np.abs(dth_deg), 95))},
        "pixel_change_pct": {"mean": float(pix_pct.mean()), "p50": float(np.median(pix_pct)), "p95": float(np.percentile(pix_pct, 95))},
        "latent_l2": {"mean": float(lat_l2.mean()), "p50": float(np.median(lat_l2)), "p95": float(np.percentile(lat_l2, 95))},
        "corr_absdx_latentL2": corr,
    }
    json.dump(stats, open(out / "chunk_motion_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nSaved -> {out}/chunk_examples.png, chunk_distributions.png, chunk_motion_stats.json")


if __name__ == "__main__":
    main()
