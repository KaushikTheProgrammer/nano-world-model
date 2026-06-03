"""
Long-horizon GT-action rollout (NanoNAV).

Autoregressively rolls the world model FAR past its H=3 training window under ground-truth actions and
decodes pred vs GT, to show how prediction holds up / compounds error over a long horizon. The rollout
chains the model's native 3-frame chunks (DiffusionWorldModel.rollout), feeding the last generated frame
as context — so the horizon is just `len(actions)`. GT frames + actions for L+1 steps are sourced by
building the val dataset with `num_frames = L+1` at the trained frame_interval.

Selects the highest-motion sequences (most total path over the L chunks) so the rollout shows real
driving/turning, not idle. Per sample writes:
  * long_<i>_grid.png   — rows [GT | Pred], cols [t0 … t+L], pred annotated with pred-vs-GT latentL2
  * long_<i>_cmp.mp4    — side-by-side GT | Pred over the full horizon
  * long_<i>_error.png  — compounding pred-vs-GT latentL2 vs rollout step

    python src/sample/long_rollout_viz.py --ckpt <...8000.ckpt> --horizon 12 --out <dir>
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

sys.path.append(os.path.split(sys.path[0])[0])

from action_diagnostic import load_checkpoint           # noqa: E402
from planning import DiffusionWorldModel                 # noqa: E402
from wm_datasets import create_train_val_datasets        # noqa: E402
from sampling_utils import decode_latents, save_comparison_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="long_rollout")
    ap.add_argument("--horizon", type=int, default=12, help="rollout steps L (each = frame_interval src frames)")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--scan-max", type=int, default=400)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
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

    L = args.horizon
    f = train_cfg.dataset.frame_interval
    H_train = train_cfg.model.num_frames - train_cfg.model.n_context_frames

    # Source L+1 frames per item (1 context + L predicted) at the trained frame_interval.
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    loader_cfg["validation_fixed_subset_size"] = None
    loader_cfg["validation_fixed_subset_path"] = None
    loader_cfg["validation_size"] = None
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=L + 1,
        frame_interval=f,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    a_mean = torch.tensor(np.asarray(val._raw_action_mean).reshape(-1)[:2], dtype=torch.float32)
    a_std = torch.tensor(np.asarray(val._raw_action_std).reshape(-1)[:2], dtype=torch.float32)

    # ---- scan for the highest-motion sequences (most total path over the L chunks) ----
    stride = max(1, len(val) // args.scan_max)
    idxs = list(range(0, len(val), stride))
    sub = torch.utils.data.Subset(val, idxs)
    scan = torch.utils.data.DataLoader(sub, batch_size=8, shuffle=False, num_workers=4)
    recs = []
    k = 0
    for batch in scan:
        act = batch["action"][:, :L].float()
        raw = act * a_std + a_mean
        dx_cm = raw[..., 0] * 100.0
        dth_deg = raw[..., 1] * (180.0 / math.pi)
        path = dx_cm.abs().sum(1) + dth_deg.abs().sum(1)
        for j in range(act.shape[0]):
            recs.append((idxs[k + j], float(path[j]), float(dx_cm[j].sum()), float(dth_deg[j].sum())))
        k += act.shape[0]
    recs.sort(key=lambda r: r[1], reverse=True)
    selected = recs[:args.num_samples]
    print(f"Scanned {len(recs)} length-{L+1} windows; selected top-{len(selected)} by motion.")

    # ---- fetch, roll out under GT actions, decode ----
    vids, acts = [], []
    for r in selected:
        item = val[r[0]]
        vids.append(item["video"])
        acts.append(item["action"])
    video = torch.stack(vids).to(device)        # [N, L+1, C, H, W]
    action = torch.stack(acts).to(device)
    N = video.shape[0]

    obs_0 = {"visual": video[:, :1]}
    act_gt = action[:, :L]
    print(f"Rolling out {N} samples, horizon L={L} ({L*f} src frames ~{L*f/30:.1f}s; "
          f"{L//H_train + (L % H_train > 0)} autoregressive chunks), {args.num_sampling_steps} DDIM steps...")
    z_obses, _ = world_model.rollout(obs_0, act_gt, num_sampling_steps=args.num_sampling_steps)
    pred_flat = z_obses["visual"]               # [N, 1+L, D]

    ref = latent_codec.encode(video[:, 0])
    C_lat, hl, wl = ref.shape[1:]
    pred_lat = pred_flat.reshape(N, 1 + L, C_lat, hl, wl)
    gt_lat = torch.stack([latent_codec.encode(video[:, t]) for t in range(1 + L)], dim=1)
    pred_frames = decode_latents(vae, pred_lat, vae_precision).clamp(0, 1).cpu().numpy()
    gt_frames = decode_latents(vae, gt_lat, vae_precision).clamp(0, 1).cpu().numpy()
    l2 = torch.norm((pred_lat - gt_lat).reshape(N, 1 + L, -1), dim=-1).cpu().numpy()

    for i, r in enumerate(selected):
        # wide grid: GT row / Pred row over the full horizon
        fig, axes = plt.subplots(2, 1 + L, figsize=(1.5 * (1 + L), 3.6))
        for t in range(1 + L):
            axes[0, t].imshow(np.transpose(gt_frames[i, t], (1, 2, 0))); axes[0, t].axis("off")
            axes[1, t].imshow(np.transpose(pred_frames[i, t], (1, 2, 0))); axes[1, t].axis("off")
            axes[0, t].set_title("GT t0" if t == 0 else f"t+{t}", fontsize=7)
            axes[1, t].set_title("Pred" if t == 0 else f"L2 {l2[i, t]:.0f}", fontsize=7)
        # mark where we pass the training horizon
        if H_train < L:
            axes[1, H_train].set_title(f"L2 {l2[i, H_train]:.0f}\n(train H)", fontsize=7, color="C3")
        fig.suptitle(f"LONG rollout L={L} ({L*f} src frames ~{L*f/30:.1f}s, {L}× steps vs train H={H_train}) — "
                     f"net Δx={r[2]:+.0f}cm Δθ={r[3]:+.0f}°  | top GT, bottom Pred", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"long_{i}_grid.png", dpi=100)
        plt.close(fig)

        save_comparison_video(
            torch.from_numpy(gt_frames[i]), torch.from_numpy(pred_frames[i]),
            str(out / f"long_{i}_cmp.mp4"), fps=args.fps,
        )

        figc, axc = plt.subplots(figsize=(5, 3))
        axc.plot(range(1, 1 + L), l2[i, 1:], "o-")
        axc.axvline(H_train + 0.5, ls=":", c="C3", label=f"train horizon H={H_train}")
        axc.set_xlabel("rollout step"); axc.set_ylabel("pred-vs-GT latentL2")
        axc.set_title(f"compounding error — sample {i} (net Δx={r[2]:+.0f}cm Δθ={r[3]:+.0f}°)")
        axc.grid(alpha=.3); axc.legend(fontsize=8)
        figc.tight_layout()
        figc.savefig(out / f"long_{i}_error.png", dpi=110)
        plt.close(figc)

    print(f"Saved {N} long rollouts -> {out}")
    for i, r in enumerate(selected):
        print(f"  sample {i}: idx={r[0]:5d}  net Δx={r[2]:+5.0f}cm Δθ={r[3]:+5.0f}°  "
              f"latentL2 step1={l2[i,1]:.1f} | trainH(step{H_train})={l2[i,H_train]:.1f} | stepL({L})={l2[i,L]:.1f}")


if __name__ == "__main__":
    main()
