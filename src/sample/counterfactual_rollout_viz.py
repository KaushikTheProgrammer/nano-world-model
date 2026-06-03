"""
Counterfactual action rollout (NanoNAV).

Every other rollout eval uses GROUND-TRUTH actions. For CEM planning to work, the world model must
produce *visibly different, action-appropriate* futures for *different* candidate actions. This script
tests exactly that: from ONE context frame it rolls out the same model under a battery of hand-specified
commands (GT / straight / turn-left / turn-right / pivot-left / pivot-right / stop), decodes them all,
and lays them side by side — plus a controllability number (how far apart the counterfactual futures
land in latent space). If "turn-left" and "turn-right" decode to the same image, no planner can succeed.

Per start frame:
  * cf_<i>_grid.png   — rows = [GT-real | one row per command], cols = [context, t+1 … t+H]
  * cf_<i>_final.png  — context + final predicted frame under each command, side by side
  * cf_<i>_<cmd>.mp4  — (optional) per-command video
Prints pairwise final-latent L2 between commands (controllability).

    python src/sample/counterfactual_rollout_viz.py --ckpt <...8000.ckpt> --horizon 5 --out <dir>
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

# Hand-specified commands in PHYSICAL units per chunk: (Δx metres, Δθ radians). f=10 full-speed Δx≈0.033 m,
# max |Δθ|≈0.14 rad (~8°). Constant over the horizon.
DEG = math.pi / 180.0
COMMANDS = {
    "straight":    (0.033,  0.0),
    "turn_left":   (0.020,  7 * DEG),
    "turn_right":  (0.020, -7 * DEG),
    "pivot_left":  (0.000,  8 * DEG),
    "pivot_right": (0.000, -8 * DEG),
    "stop":        (0.000,  0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="counterfactual")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num-starts", type=int, default=3)
    ap.add_argument("--scan-max", type=int, default=300)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--videos", action="store_true", help="also write per-command mp4s")
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

    H = args.horizon
    f = train_cfg.dataset.frame_interval

    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    loader_cfg["validation_fixed_subset_size"] = None
    loader_cfg["validation_fixed_subset_path"] = None
    loader_cfg["validation_size"] = None
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=H + 1,
        frame_interval=f,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    a_mean = torch.tensor(np.asarray(val._raw_action_mean).reshape(-1)[:2], dtype=torch.float32)
    a_std = torch.tensor(np.asarray(val._raw_action_std).reshape(-1)[:2], dtype=torch.float32)

    def normalize(dx_m, dth_rad):
        raw = torch.tensor([dx_m, dth_rad], dtype=torch.float32)
        return (raw - a_mean) / a_std

    # pick start frames that are "driving" contexts (GT forward motion -> open space ahead)
    stride = max(1, len(val) // args.scan_max)
    idxs = list(range(0, len(val), stride))
    sub = torch.utils.data.Subset(val, idxs)
    scan = torch.utils.data.DataLoader(sub, batch_size=8, shuffle=False, num_workers=4)
    recs, k = [], 0
    for batch in scan:
        raw = batch["action"][:, :H].float() * a_std + a_mean
        fwd = (raw[..., 0] * 100.0).sum(1)  # net forward cm
        for j in range(raw.shape[0]):
            recs.append((idxs[k + j], float(fwd[j])))
        k += raw.shape[0]
    recs.sort(key=lambda r: r[1], reverse=True)
    starts = [r[0] for r in recs[:args.num_starts]]
    print(f"Start frames (driving contexts): {starts}")

    cmd_names = ["gt"] + list(COMMANDS.keys())

    for si, sidx in enumerate(starts):
        item = val[sidx]
        ctx = item["video"][:1].to(device)            # [1, C, H, W]
        gt_video = item["video"].to(device)           # [H+1, C, H, W]
        gt_act = item["action"][:H].to(device)        # [H, 2] (normalized)

        # build action batch: row 0 = GT, then each command (constant over H)
        acts = [gt_act]
        for name in COMMANDS:
            dx, dth = COMMANDS[name]
            a = normalize(dx, dth).to(device).unsqueeze(0).repeat(H, 1)  # [H, 2]
            acts.append(a)
        action = torch.stack(acts, dim=0)              # [n_cmd, H, 2]
        n_cmd = action.shape[0]
        obs_0 = {"visual": ctx.unsqueeze(0).repeat(n_cmd, 1, 1, 1, 1)}  # [n_cmd, 1, C, H, W]

        z_obses, _ = world_model.rollout(obs_0, action, num_sampling_steps=args.num_sampling_steps)
        pred_flat = z_obses["visual"]                  # [n_cmd, 1+H, D]
        ref = latent_codec.encode(ctx); C_lat, hl, wl = ref.shape[1:]
        pred_lat = pred_flat.reshape(n_cmd, 1 + H, C_lat, hl, wl)
        pred_frames = decode_latents(vae, pred_lat, vae_precision).clamp(0, 1).cpu().numpy()  # [n_cmd,1+H,...]

        # GT-real reference row (VAE round-trip of the true frames)
        gt_lat = torch.stack([latent_codec.encode(gt_video[t:t+1]) for t in range(H + 1)], dim=1)  # [1,H+1,...]
        gt_real = decode_latents(vae, gt_lat, vae_precision).clamp(0, 1).cpu().numpy()[0]          # [H+1,...]

        # ---- grid: GT-real + one row per command ----
        rows = 1 + n_cmd
        fig, axes = plt.subplots(rows, 1 + H, figsize=(1.8 * (1 + H), 1.7 * rows))
        titles = ["context"] + [f"t+{t}" for t in range(1, 1 + H)]
        for t in range(1 + H):
            axes[0, t].imshow(np.transpose(gt_real[t], (1, 2, 0))); axes[0, t].axis("off")
            axes[0, t].set_title(("GT-real " if t == 0 else "") + titles[t], fontsize=8)
        for r, name in enumerate(cmd_names):
            for t in range(1 + H):
                axes[r + 1, t].imshow(np.transpose(pred_frames[r, t], (1, 2, 0))); axes[r + 1, t].axis("off")
            if name == "gt":
                lab = "PRED: gt actions"
            else:
                dx, dth = COMMANDS[name]
                lab = f"PRED: {name}\nΔx={dx*100:.0f}cm Δθ={dth/DEG:+.0f}°/step"
            axes[r + 1, 0].set_title(lab, fontsize=8)
        fig.suptitle(f"Counterfactual rollout — start idx {sidx}, H={H} ({H*f} src frames ~{H*f/30:.1f}s). "
                     f"Same context, different commands. (step-8000)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"cf_{si}_grid.png", dpi=100)
        plt.close(fig)

        # ---- final-frame comparison strip ----
        figf, axf = plt.subplots(1, 1 + n_cmd, figsize=(2.0 * (1 + n_cmd), 2.4))
        axf[0].imshow(np.transpose(pred_frames[0, 0], (1, 2, 0))); axf[0].axis("off"); axf[0].set_title("context", fontsize=9)
        for r, name in enumerate(cmd_names):
            axf[r + 1].imshow(np.transpose(pred_frames[r, -1], (1, 2, 0))); axf[r + 1].axis("off")
            axf[r + 1].set_title(name, fontsize=9)
        figf.suptitle(f"Final predicted frame (t+{H}) under each command — start idx {sidx}", fontsize=10)
        figf.tight_layout(); figf.savefig(out / f"cf_{si}_final.png", dpi=110); plt.close(figf)

        if args.videos:
            for r, name in enumerate(cmd_names):
                save_comparison_video(torch.from_numpy(gt_real), torch.from_numpy(pred_frames[r]),
                                      str(out / f"cf_{si}_{name}.mp4"), fps=args.fps)

        # ---- controllability: pairwise final-latent L2 between commands ----
        finals = pred_lat[:, -1].reshape(n_cmd, -1)
        def l2(a, b): return float(torch.norm(finals[cmd_names.index(a)] - finals[cmd_names.index(b)]))
        print(f"\nstart idx {sidx} — final-latent L2 between commands (bigger = more controllable):")
        print(f"  turn_left vs turn_right : {l2('turn_left','turn_right'):.1f}")
        print(f"  pivot_left vs pivot_right: {l2('pivot_left','pivot_right'):.1f}")
        print(f"  straight vs stop         : {l2('straight','stop'):.1f}")
        print(f"  straight vs pivot_left   : {l2('straight','pivot_left'):.1f}")
        print(f"  gt vs stop               : {l2('gt','stop'):.1f}")

    print(f"\nSaved counterfactual grids -> {out}")


if __name__ == "__main__":
    main()
