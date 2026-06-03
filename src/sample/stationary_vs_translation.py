"""
Stationary vs pure-translation SD-VAE latent comparison (NanoNAV).

The f-sweep showed corr(|Δx|, latentL2) ≈ 0, but that statistic lumps stationary, translation,
rotation and arc chunks together. This script runs the *sharp, controlled* test:

    Does a PURE-TRANSLATION chunk change the SD-VAE latent any more than a STATIONARY chunk?

If the two latentL2 distributions overlap, forward motion is provably below the non-action latent
noise floor (lighting flicker, codec, sensor noise) — i.e. there is no signal for the action branch
to ground Δx on. This is the cleanest evidence for the translation-observability diagnosis.

Chunk classes (per consecutive frame pair k, k+f — exactly the pairs the WM is trained on):
  * STATIONARY        : |Δx| < dx_eps  AND |Δθ| < dth_eps     (robot not commanded to move)
  * PURE-TRANSLATION  : |Δx| > dx_hi   AND |Δθ| < dth_eps     (driving straight, no yaw)
  * PURE-ROTATION     : |Δθ| > dth_hi  AND |Δx| < dx_eps      (turning in place — the positive control)

For each class we compare:
  * latentL2 = ||z(k+f) - z(k)||_F  distribution (mean/median/std, histogram overlay)
  * effect of translation over the noise floor:
      SNR    = (mean_L2[trans] - mean_L2[stat]) / std_L2[stat]
      Cohen d= (mean_L2[trans] - mean_L2[stat]) / pooled_std
      AUC    = P(L2 of a random trans chunk > L2 of a random stat chunk)  (0.5 == indistinguishable)
  * mean per-cell ||Δz|| spatial map per class, and (trans - stat) difference map — shows *where*
    (if anywhere) forward motion leaves a latent footprint (e.g. near-field floor).

Output (to --out):
  * latent_compare.png        — latentL2 histogram overlay + mean spatial maps (stat | trans | diff | rot)
  * latent_compare_stats.json — all numbers

    python src/sample/stationary_vs_translation.py --ckpt <run>/checkpoints/.../X.ckpt --out <dir>
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


def mannwhitney_auc(a, b):
    """AUC = P(x in `a` > y in `b`), ties counted as 0.5. Equivalent to U/(na*nb).
    Computed via rank sums so it is O((na+nb) log(na+nb))."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    Ra = ranks[:na].sum()
    Ua = Ra - na * (na + 1) / 2.0
    return float(Ua / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="any run checkpoint (used only for its SD-VAE + config)")
    ap.add_argument("--out", default="latent_compare")
    ap.add_argument("--n-batches", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--frame-interval", type=int, default=None,
                    help="override the run's frame_interval to preview a larger chunk (more Δx per chunk) "
                         "WITHOUT retraining (SD-VAE+config only). e.g. 20.")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for which scenes are scanned")
    ap.add_argument("--dx-eps", type=float, default=0.3, help="|Δx|<this (cm) counts as stationary in x")
    ap.add_argument("--dth-eps", type=float, default=0.5, help="|Δθ|<this (deg) counts as no-rotation")
    ap.add_argument("--dx-hi", type=float, default=1.3, help="|Δx|>this (cm) counts as translating")
    ap.add_argument("--dth-hi", type=float, default=4.0, help="|Δθ|>this (deg) counts as rotating")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, latent_codec, _, train_cfg = load_checkpoint(args.ckpt, device)
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    fi = args.frame_interval if args.frame_interval is not None else train_cfg.dataset.frame_interval
    chunk_ms = fi * 1000.0 / 30.0
    print(f"Comparing stationary vs pure-translation at frame_interval={fi} ({chunk_ms:.0f} ms/chunk)"
          f"{'' if args.frame_interval is None else ' [OVERRIDE]'}")

    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=fi,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    a_mean = np.asarray(val._raw_action_mean).reshape(-1)[:2]   # (Δx [m], Δθ [rad])
    a_std = np.asarray(val._raw_action_std).reshape(-1)[:2]
    g = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, generator=g)

    # per-class accumulators
    classes = ["stationary", "translation", "rotation"]
    l2 = {c: [] for c in classes}
    dzsum = {c: None for c in classes}   # running sum of per-cell ||Δz|| maps -> mean later
    dzn = {c: 0 for c in classes}
    all_dx, all_dth, all_l2 = [], [], []

    n = 0
    for batch in loader:
        if n >= args.n_batches:
            break
        video = batch["video"].to(device)        # [B, 1+H, C, H, W], pixels in [-1,1]
        action = batch["action"].cpu().numpy()    # [B, 1+H, 2] normalized
        B, T = video.shape[:2]
        for t in range(T - 1):
            z0 = latent_codec.encode(video[:, t])
            z1 = latent_codec.encode(video[:, t + 1])
            dz_map = torch.norm(z1 - z0, dim=1).cpu().numpy()        # [B, h, w]
            l2b = np.linalg.norm(dz_map.reshape(B, -1), axis=-1)     # ||Δz||_F
            act_phys = action[:, t] * a_std + a_mean
            dxc = np.abs(act_phys[:, 0] * 100.0)                     # |Δx| cm
            dthd = np.abs(np.degrees(act_phys[:, 1]))               # |Δθ| deg
            all_dx.extend(dxc.tolist()); all_dth.extend(dthd.tolist()); all_l2.extend(l2b.tolist())
            for b in range(B):
                if dthd[b] < args.dth_eps and dxc[b] < args.dx_eps:
                    c = "stationary"
                elif dthd[b] < args.dth_eps and dxc[b] > args.dx_hi:
                    c = "translation"
                elif dxc[b] < args.dx_eps and dthd[b] > args.dth_hi:
                    c = "rotation"
                else:
                    continue
                l2[c].append(float(l2b[b]))
                dzsum[c] = dz_map[b].copy() if dzsum[c] is None else dzsum[c] + dz_map[b]
                dzn[c] += 1
        n += 1

    for c in classes:
        l2[c] = np.asarray(l2[c])
    s, tr, ro = l2["stationary"], l2["translation"], l2["rotation"]

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    def sd(x): return float(np.std(x)) if len(x) else float("nan")
    pooled = np.sqrt((sd(s) ** 2 + sd(tr) ** 2) / 2.0) if len(s) and len(tr) else float("nan")
    snr = (m(tr) - m(s)) / sd(s) if len(s) else float("nan")
    cohen = (m(tr) - m(s)) / pooled if pooled and not np.isnan(pooled) else float("nan")
    auc_tr = mannwhitney_auc(tr, s)     # P(translation L2 > stationary L2)
    auc_ro = mannwhitney_auc(ro, s)     # positive control

    stats = {
        "frame_interval": int(fi), "chunk_ms": float(chunk_ms),
        "thresholds_cm_deg": {"dx_eps": args.dx_eps, "dth_eps": args.dth_eps,
                              "dx_hi": args.dx_hi, "dth_hi": args.dth_hi},
        "n": {c: int(len(l2[c])) for c in classes},
        "latentL2": {c: {"mean": m(l2[c]), "median": float(np.median(l2[c])) if len(l2[c]) else float("nan"),
                         "std": sd(l2[c])} for c in classes},
        "translation_vs_stationary": {
            "delta_mean_L2": m(tr) - m(s),
            "SNR_over_noise_floor": snr,        # (mean_t - mean_s) / std_s
            "cohens_d": cohen,
            "AUC_trans_gt_stat": auc_tr,        # 0.5 == indistinguishable
        },
        "rotation_vs_stationary_control": {
            "delta_mean_L2": m(ro) - m(s),
            "cohens_d": (m(ro) - m(s)) / (np.sqrt((sd(s) ** 2 + sd(ro) ** 2) / 2.0)) if len(s) and len(ro) else float("nan"),
            "AUC_rot_gt_stat": auc_ro,
        },
    }
    json.dump(stats, open(out / "latent_compare_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

    # ---------- figure ----------
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    # (top, spanning) histogram overlay of latentL2
    axh = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, max(np.percentile(all_l2, 99.5), 1.0), 70)
    axh.hist(s, bins=bins, density=True, alpha=0.55, color="gray", label=f"stationary (n={len(s)}, μ={m(s):.1f})")
    axh.hist(tr, bins=bins, density=True, alpha=0.55, color="steelblue",
             label=f"pure-translation (n={len(tr)}, μ={m(tr):.1f})")
    axh.hist(ro, bins=bins, density=True, alpha=0.45, color="slateblue",
             label=f"pure-rotation [control] (n={len(ro)}, μ={m(ro):.1f})")
    axh.axvline(m(s), color="gray", ls="--"); axh.axvline(m(tr), color="steelblue", ls="--")
    axh.set_title(f"SD-VAE latentL2 per chunk by motion class (f={fi}, {chunk_ms:.0f} ms)\n"
                  f"translation vs stationary: SNR={snr:.2f}σ, Cohen d={cohen:.2f}, "
                  f"AUC={auc_tr:.3f} (0.5=indistinguishable)  |  rotation control AUC={auc_ro:.3f}")
    axh.set_xlabel("latentL2 = ||z(k+f) − z(k)||_F"); axh.set_ylabel("density"); axh.legend()

    # (bottom) mean per-cell ||Δz|| maps: stationary | translation | (trans - stat) | rotation
    means = {c: (dzsum[c] / dzn[c] if dzn[c] else None) for c in classes}
    vmax = max((np.percentile(means[c], 99) for c in classes if means[c] is not None), default=1.0)
    panels = [("stationary", means["stationary"], "inferno", 0, vmax),
              ("translation", means["translation"], "inferno", 0, vmax)]
    if means["stationary"] is not None and means["translation"] is not None:
        diff = means["translation"] - means["stationary"]
        dmax = float(np.abs(diff).max()) or 1e-6
        panels.append(("translation − stationary", diff, "coolwarm", -dmax, dmax))
    else:
        panels.append(("translation − stationary", None, "coolwarm", -1, 1))
    panels.append(("rotation [control]", means["rotation"], "inferno", 0, vmax))
    for i, (title, arr, cmap, vmn, vmx) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        if arr is not None:
            im = ax.imshow(arr, cmap=cmap, vmin=vmn, vmax=vmx, interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"mean |Δz|: {title}", fontsize=9); ax.axis("off")
    fig.suptitle("Where (if anywhere) does forward motion leave a latent footprint?", y=0.995, fontsize=11)
    fig.tight_layout(); fig.savefig(out / "latent_compare.png", dpi=120); plt.close(fig)
    print(f"\nSaved -> {out}/latent_compare.png, latent_compare_stats.json")


if __name__ == "__main__":
    main()
