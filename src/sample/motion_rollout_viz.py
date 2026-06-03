"""
Motion-selected GT-action rollout visualization (NanoNAV).

`gt_rollout_viz.py` just takes the first val batch, which (with bang-bang LeKiwi data) is mostly
STATIONARY chunks — useless for judging whether the world model predicts real motion. This script
instead SCANS the val set, classifies each chunk by its integrated GT motion (Δx, Δθ), and rolls out
the highest-motion examples in three buckets so you can see prediction vs ground truth for REAL motion:

  * translation  — large net |Δx|, near-zero rotation   (robot drives forward)
  * rotation     — large net |Δθ|, near-zero translation (robot turns in place)
  * arc          — both large                            (drive + turn)

Per selected chunk it writes (same format/decoder as gt_rollout_viz, so GT and Pred share the VAE
round-trip path — isolating world-model error from VAE error):
  * <cat>_<i>_grid.png  — rows [GT | Pred], cols [context, t+1 … t+H], annotated with per-step
                          Δx (cm) / Δθ (deg) and pred-vs-GT latentL2
  * <cat>_<i>_cmp.mp4    — side-by-side GT | Pred over time

    python src/sample/motion_rollout_viz.py --ckpt <run>/checkpoints/.../X.ckpt --out <run>/motion_rollout
"""
import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])  # put src/ on the path

from action_diagnostic import load_checkpoint           # noqa: E402
from planning import DiffusionWorldModel                 # noqa: E402
from wm_datasets import create_train_val_datasets        # noqa: E402
from sampling_utils import decode_latents, save_comparison_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="motion_rollout")
    ap.add_argument("--scan-max", type=int, default=640, help="how many strided val slices to scan for motion")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42, help="seed diffusion noise for fair cross-checkpoint comparison")
    # selection thresholds (physical units; defaults tuned for f=10)
    ap.add_argument("--trans-dth-max", type=float, default=3.0, help="max |net Δθ| (deg) for a translation chunk")
    ap.add_argument("--rot-dx-max", type=float, default=2.0, help="max |net Δx| (cm) for a rotation chunk")
    ap.add_argument("--arc-dx-min", type=float, default=4.0)
    ap.add_argument("--arc-dth-min", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, latent_codec, diffusion, train_cfg = load_checkpoint(args.ckpt, device)
    wm_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    world_model = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)
    vae = latent_codec.vae
    vae_precision = getattr(latent_codec, "precision", "fp32")

    H = train_cfg.model.num_frames - train_cfg.model.n_context_frames

    # Use the FULL val set (not the 256-window fixed monitor subset) so there's a rich motion pool.
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    loader_cfg["validation_fixed_subset_size"] = None
    loader_cfg["validation_fixed_subset_path"] = None
    loader_cfg["validation_size"] = None
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=train_cfg.dataset.frame_interval,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )

    a_mean = torch.tensor(np.asarray(val._raw_action_mean).reshape(-1)[:2], dtype=torch.float32)
    a_std = torch.tensor(np.asarray(val._raw_action_std).reshape(-1)[:2], dtype=torch.float32)

    # ---- scan a strided slice of the val set for per-chunk motion (covers all episodes evenly) ----
    stride = max(1, len(val) // args.scan_max)
    scan_indices = list(range(0, len(val), stride))
    sub = torch.utils.data.Subset(val, scan_indices)
    scan_loader = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=False, num_workers=4)

    recs = []  # (idx, net_dx_cm, net_dth_deg, [Δx per step cm], [Δθ per step deg])
    k = 0
    for batch in scan_loader:
        act = batch["action"][:, :H].float()       # [B, H, 2] normalized
        raw = act * a_std + a_mean                  # [B, H, 2] -> (m, rad)
        dx_cm = raw[..., 0] * 100.0                 # [B, H]
        dth_deg = raw[..., 1] * (180.0 / math.pi)   # [B, H]
        net_dx = dx_cm.sum(1)
        net_dth = dth_deg.sum(1)
        B = act.shape[0]
        for j in range(B):
            recs.append((scan_indices[k + j], float(net_dx[j]), float(net_dth[j]),
                         [float(v) for v in dx_cm[j]], [float(v) for v in dth_deg[j]]))
        k += B
    print(f"Scanned {len(recs)} val chunks (stride {stride}).")

    def pick(cond, key, n):
        cand = [r for r in recs if cond(r)]
        cand.sort(key=key, reverse=True)
        return cand[:n]

    K = args.per_category
    translation = pick(lambda r: abs(r[2]) < args.trans_dth_max, lambda r: abs(r[1]), K)
    rotation = pick(lambda r: abs(r[1]) < args.rot_dx_max, lambda r: abs(r[2]), K)
    arc = pick(lambda r: abs(r[1]) > args.arc_dx_min and abs(r[2]) > args.arc_dth_min,
               lambda r: abs(r[1]) + abs(r[2]), K)
    selected = ([("translation", r) for r in translation]
                + [("rotation", r) for r in rotation]
                + [("arc", r) for r in arc])
    if not selected:
        print("No high-motion chunks found; loosen thresholds or raise --scan-max.")
        return

    # ---- fetch the selected chunks, roll out under GT actions, decode ----
    vids, acts = [], []
    for _, r in selected:
        item = val[r[0]]
        vids.append(item["video"])
        acts.append(item["action"])
    video = torch.stack(vids).to(device)            # [N, 1+H, C, H, W]
    action = torch.stack(acts).to(device)
    N = video.shape[0]

    obs_0 = {"visual": video[:, :1]}
    act_gt = action[:, :H]
    print(f"Rolling out {N} motion chunks (H={H}, {args.num_sampling_steps} DDIM steps)...")
    z_obses, _ = world_model.rollout(obs_0, act_gt, num_sampling_steps=args.num_sampling_steps)
    pred_flat = z_obses["visual"]

    ref = latent_codec.encode(video[:, 0])
    C_lat, hl, wl = ref.shape[1:]
    pred_lat = pred_flat.reshape(N, 1 + H, C_lat, hl, wl)
    gt_lat = torch.stack([latent_codec.encode(video[:, t]) for t in range(1 + H)], dim=1)
    pred_frames = decode_latents(vae, pred_lat, vae_precision).clamp(0, 1).cpu().numpy()
    gt_frames = decode_latents(vae, gt_lat, vae_precision).clamp(0, 1).cpu().numpy()
    l2 = torch.norm((pred_lat - gt_lat).reshape(N, 1 + H, -1), dim=-1).cpu().numpy()

    titles = ["context"] + [f"t+{i}" for i in range(1, 1 + H)]
    cat_counts = {}
    for i, (cat, r) in enumerate(selected):
        ci = cat_counts.get(cat, 0)
        cat_counts[cat] = ci + 1
        dxs, dths = r[3], r[4]
        fig, axes = plt.subplots(2, 1 + H, figsize=(3 * (1 + H), 6))
        for t in range(1 + H):
            axes[0, t].imshow(np.transpose(gt_frames[i, t], (1, 2, 0)))
            axes[0, t].set_title(f"GT {titles[t]}")
            axes[0, t].axis("off")
            cap = f"Pred {titles[t]}"
            if t > 0:
                cap += f"\nΔx={dxs[t-1]:+.1f}cm Δθ={dths[t-1]:+.1f}°\nlatL2={l2[i, t]:.1f}"
            axes[1, t].imshow(np.transpose(pred_frames[i, t], (1, 2, 0)))
            axes[1, t].set_title(cap, fontsize=8)
            axes[1, t].axis("off")
        fig.suptitle(f"{cat.upper()} — net Δx={r[1]:+.1f}cm, Δθ={r[2]:+.1f}°  "
                     f"(top: GT via VAE, bottom: world-model prediction)")
        fig.tight_layout()
        fig.savefig(out / f"{cat}_{ci}_grid.png", dpi=110)
        plt.close(fig)
        save_comparison_video(
            torch.from_numpy(gt_frames[i]), torch.from_numpy(pred_frames[i]),
            str(out / f"{cat}_{ci}_cmp.mp4"), fps=args.fps,
        )

    print(f"Saved {N} motion rollouts -> {out}")
    for i, (cat, r) in enumerate(selected):
        print(f"  {cat:11s} idx={r[0]:5d}  net Δx={r[1]:+6.1f}cm  net Δθ={r[2]:+6.1f}°  "
              f"meanLatL2={float(np.mean(l2[i, 1:])):.1f}")


if __name__ == "__main__":
    main()
