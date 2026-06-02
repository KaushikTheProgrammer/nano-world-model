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
    ap.add_argument("--frame-interval", type=int, default=None,
                    help="override the run's frame_interval to preview a different chunk size "
                         "(e.g. 10). The checkpoint is only used for its SD-VAE+config, so this "
                         "previews the training signal at a candidate f WITHOUT retraining.")
    ap.add_argument("--seed", type=int, default=None,
                    help="shuffle the val loader with this seed so the example chunks are drawn from "
                         "different scenes/episodes. Default (None) = deterministic in-order draw. "
                         "Distribution stats are unaffected by which chunks are shown.")
    ap.add_argument("--example-mode", choices=["mixed", "forward", "rotate", "arc"], default="mixed",
                    help="which chunks to show as examples: 'forward' (max |Δx|), 'rotate' (max |Δθ|), "
                         "'arc' (both — driving while turning), or 'mixed' (default; a spread of all "
                         "three). Examples are de-duplicated by episode+time so each row is a distinct "
                         "moment, not an overlapping stride-1 neighbour.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, latent_codec, _, train_cfg = load_checkpoint(args.ckpt, device)

    # un-normalization stats for the integrated action (set when the dataset is built)
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    fi = args.frame_interval if args.frame_interval is not None else train_cfg.dataset.frame_interval
    chunk_ms = fi * 1000.0 / 30.0  # source data is 30 Hz
    print(f"Previewing per-chunk signal at frame_interval={fi} ({chunk_ms:.0f} ms/chunk)"
          f"{' [run default]' if args.frame_interval is None else ' [OVERRIDE]'}")
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=fi,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    a_mean = np.asarray(val._raw_action_mean).reshape(-1)[:2]   # (Δx [m], Δθ [rad])
    a_std = np.asarray(val._raw_action_std).reshape(-1)[:2]
    if args.seed is not None:
        g = torch.Generator().manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, generator=g)
    else:
        loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dx_cm, dth_deg, pix_pct, lat_l2 = [], [], [], []
    # Candidate example chunks (collected at t==0 only, so each maps to one window with a clean
    # (traj_idx, start_idx) identity for de-duplication). Each entry is a dict carrying the frames
    # and per-chunk metrics; selection by --example-mode happens after the scan.
    cands = []
    gap = max(fi, 15)   # min frame separation (within an episode) for two examples to count as distinct

    n = 0
    for batch in loader:
        if n >= args.n_batches:
            break
        video = batch["video"].to(device)        # [B, 1+H, C, H, W], pixels in [-1,1]
        action = batch["action"].cpu().numpy()    # [B, 1+H, 2] normalized
        traj_b = np.asarray(batch["meta_info"]["traj_idx"]).reshape(-1)
        start_b = np.asarray(batch["meta_info"]["start_idx"]).reshape(-1)
        B, T = video.shape[:2]
        with torch.no_grad():
            for t in range(T - 1):                # each consecutive pair = one chunk
                z0 = latent_codec.encode(video[:, t])
                z1 = latent_codec.encode(video[:, t + 1])
                # Per-spatial-cell SD-VAE latent change ||Δz||_2 over the 4 channels -> [B, h, w].
                # This (not the pixel diff) is what the world model is trained to predict (v-pred in
                # latent space), so latL2 = its Frobenius norm is the signal the action branch sees.
                dz_map = torch.norm(z1 - z0, dim=1).cpu().numpy()          # [B, h, w]
                l2 = np.linalg.norm(dz_map.reshape(B, -1), axis=-1)         # == ||z1 - z0||_F
                f0 = ((video[:, t].clamp(-1, 1) + 1) / 2)
                f1 = ((video[:, t + 1].clamp(-1, 1) + 1) / 2)
                pix = (f1 - f0).abs().mean(dim=(1, 2, 3)).cpu().numpy() * 100.0  # % of full range (reference only)
                act_phys = action[:, t] * a_std + a_mean      # [B,2] -> (Δx m, Δθ rad)
                dxc = act_phys[:, 0] * 100.0
                dthd = np.degrees(act_phys[:, 1])
                dx_cm.extend(dxc.tolist()); dth_deg.extend(dthd.tolist())
                pix_pct.extend(pix.tolist()); lat_l2.extend(l2.tolist())
                if t == 0:   # collect example candidates only from each window's leading chunk
                    for b in range(B):
                        cands.append(dict(
                            traj=int(traj_b[b]), start=int(start_b[b]),
                            dx=float(dxc[b]), dth=float(dthd[b]), pix=float(pix[b]), l2=float(l2[b]),
                            f0=f0[b].cpu().numpy(), f1=f1[b].cpu().numpy(), dz=dz_map[b]))
        n += 1

    dx_cm, dth_deg, pix_pct, lat_l2 = map(np.asarray, (dx_cm, dth_deg, pix_pct, lat_l2))

    # ---- pick DIVERSE examples (distinct episode+time), by motion type ----
    def far(e, chosen):  # True if e is a distinct moment from everything already chosen
        return all(not (e["traj"] == c["traj"] and abs(e["start"] - c["start"]) < gap) for c in chosen)

    def take(score, chosen, k):
        # greedily take up to k highest-score candidates that are distinct from `chosen`
        added = 0
        for e in sorted(cands, key=score, reverse=True):
            if added >= k:
                break
            if e in chosen or not far(e, chosen):
                continue
            chosen.append(e); added += 1
        return added

    s_fwd = lambda e: abs(e["dx"])
    s_rot = lambda e: abs(e["dth"])
    s_arc = lambda e: abs(e["dx"]) * abs(e["dth"])   # both must be substantial
    labels = {id(e): "" for e in cands}
    chosen = []
    if args.example_mode == "forward":
        take(s_fwd, chosen, args.n_examples); cat = {id(e): "drive" for e in chosen}
    elif args.example_mode == "rotate":
        take(s_rot, chosen, args.n_examples); cat = {id(e): "rotate" for e in chosen}
    elif args.example_mode == "arc":
        take(s_arc, chosen, args.n_examples); cat = {id(e): "drive+turn" for e in chosen}
    else:  # mixed: a spread of drive / rotate / drive+turn
        cat = {}
        per = max(1, args.n_examples // 3)
        for score, name in [(s_fwd, "drive"), (s_rot, "rotate"), (s_arc, "drive+turn")]:
            before = len(chosen); take(score, chosen, per)
            for e in chosen[before:]:
                cat[id(e)] = name
        take(s_fwd, chosen, args.n_examples - len(chosen))   # backfill if a category was thin
        for e in chosen:
            cat.setdefault(id(e), "drive")
    chosen = chosen[:args.n_examples]

    # ---- examples montage: frame k | frame k+f | SD-VAE latent change map ----
    ne = len(chosen)
    # Shared color scale (robust max over rows) so latent-change magnitudes are comparable across rows.
    dz_vmax = max((np.percentile(e["dz"], 99) for e in chosen), default=1e-3)
    fig, axes = plt.subplots(ne, 3, figsize=(10.5, 3 * ne))
    if ne == 1:
        axes = axes[None, :]
    for i, e in enumerate(chosen):
        axes[i, 0].imshow(np.transpose(e["f0"], (1, 2, 0)))
        axes[i, 0].set_title(f"[{cat.get(id(e),'')}]  frame k\nep{e['traj']} @{e['start']}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(np.transpose(e["f1"], (1, 2, 0)))
        axes[i, 1].set_title(f"frame k+{fi}  (Δx={e['dx']:.2f}cm, Δθ={e['dth']:.1f}°)"); axes[i, 1].axis("off")
        # 32x32 per-cell ||Δz|| over the 4 SD-VAE channels — the change the model actually predicts.
        im = axes[i, 2].imshow(e["dz"], cmap="inferno", vmin=0, vmax=dz_vmax, interpolation="nearest")
        axes[i, 2].set_title(f"SD-VAE |Δz| per cell  (latL2={e['l2']:.1f})"); axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046)
    fig.suptitle(f"Per-chunk (f={fi}, {chunk_ms:.0f} ms) SD-VAE latent change — mode={args.example_mode} "
                 f"(rows are distinct episode+time moments)")
    fig.tight_layout(); fig.savefig(out / "chunk_examples.png", dpi=110); plt.close(fig)

    # ---- distributions ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(np.abs(dx_cm), bins=60, color="steelblue"); ax[0, 0].set_title("|Δx| per chunk (cm) — translation"); ax[0, 0].set_xlabel("cm")
    ax[0, 1].hist(np.abs(dth_deg), bins=60, color="slateblue"); ax[0, 1].set_title("|Δθ| per chunk (deg) — rotation"); ax[0, 1].set_xlabel("deg")
    ax[0, 2].hist(pix_pct, bins=60, color="seagreen"); ax[0, 2].set_title("mean |Δpixel| per chunk (% of range)"); ax[0, 2].set_xlabel("%")
    ax[1, 0].hist(lat_l2, bins=60, color="darkred"); ax[1, 0].set_title("SD-VAE latent L2 per chunk (the action signal)"); ax[1, 0].set_xlabel("L2")
    corr_dx = float(np.corrcoef(np.abs(dx_cm), lat_l2)[0, 1])
    corr_dth = float(np.corrcoef(np.abs(dth_deg), lat_l2)[0, 1])
    ax[1, 1].scatter(np.abs(dx_cm), lat_l2, s=4, alpha=0.3, color="steelblue")
    ax[1, 1].set_title(f"translation vs latent change  (corr={corr_dx:.2f})"); ax[1, 1].set_xlabel("|Δx| (cm)"); ax[1, 1].set_ylabel("latent L2")
    ax[1, 2].scatter(np.abs(dth_deg), lat_l2, s=4, alpha=0.3, color="slateblue")
    ax[1, 2].set_title(f"rotation vs latent change  (corr={corr_dth:.2f})"); ax[1, 2].set_xlabel("|Δθ| (deg)"); ax[1, 2].set_ylabel("latent L2")
    fig.suptitle(f"Per-chunk motion vs frame/latent change (f={fi}) — which action component does the camera actually see?")
    fig.tight_layout(); fig.savefig(out / "chunk_distributions.png", dpi=110); plt.close(fig)

    stats = {
        "n_chunks": int(len(dx_cm)),
        "dx_cm": {"mean": float(np.abs(dx_cm).mean()), "p50": float(np.median(np.abs(dx_cm))), "p95": float(np.percentile(np.abs(dx_cm), 95)), "max": float(np.abs(dx_cm).max())},
        "dtheta_deg": {"mean_abs": float(np.abs(dth_deg).mean()), "p95": float(np.percentile(np.abs(dth_deg), 95))},
        "pixel_change_pct": {"mean": float(pix_pct.mean()), "p50": float(np.median(pix_pct)), "p95": float(np.percentile(pix_pct, 95))},
        "latent_l2": {"mean": float(lat_l2.mean()), "p50": float(np.median(lat_l2)), "p95": float(np.percentile(lat_l2, 95))},
        "corr_absdx_latentL2": corr_dx,
        "corr_absdth_latentL2": corr_dth,
        "frame_interval": int(fi),
        "chunk_ms": float(chunk_ms),
    }
    json.dump(stats, open(out / "chunk_motion_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nSaved -> {out}/chunk_examples.png, chunk_distributions.png, chunk_motion_stats.json")


if __name__ == "__main__":
    main()
