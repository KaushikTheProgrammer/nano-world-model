"""
Offline CEM planning eval (NanoNAV Stage 6a).

The one question 6a answers: *given a goal image, can the planner recover steering commands that reach
it* — and does that hold at the cheap sampler settings (DDIM=3) that make 6b's ~7 s/replan viable?

This is a STANDALONE eval (not a registry env): LeKiwi has no simulator and no way to execute an
arbitrary CEM action offline, so we do NOT fake a LeKiwi env (its step() would have to use the WM as
its own dynamics — circular). Instead we follow the Run-002 eval-tool pattern (motion_rollout_viz.py /
long_rollout_viz.py): load ckpt + dataset directly, run the REAL CEMPlanner + DiffusionWorldModel, and
grade against the dataset as a built-in answer key.

Per held-out val scene we take a start frame z0 and a goal frame `goal_H` chunks ahead (a real,
reachable goal whose TRUE commands we know). We hide the commands, give CEM only the two frames, let it
plan (sample command sequences -> WM rollout -> keep the ones whose imagined future looks most like the
goal -> refit), and grade:

  do_nothing      = L2(z0, z_goal)                          floor CEM must beat (never move)
  gt_ceiling      = L2(WM.rollout(z0, GT actions)_-1, z_goal)   WM accuracy ceiling (CEM can't beat WM error)
  cem_reached     = L2(WM.rollout(z0, CEM actions)_-1, z_goal)  did CEM drive the WM to the goal?
  action_recovery = denorm(CEM a) vs GT (Dx,Dtheta)            strongest offline signal: re-derived commands?
  decoded montage = z0 / goal / CEM-planned rollout via SD-VAE  eyeball whether planned future ~ goal

All L2 are latent-space norms (same convention as motion_rollout_viz, so numbers are comparable ~30).
What this measures = OPEN-LOOP planning accuracy (a necessary precondition), NOT closed-loop success
(that's 6b — there is no offline ground truth for where the robot ends up if it executes CEM's actions).

Headline deliverable: the whole battery is swept over DDIM in {20, 5, 3}. If DDIM=3 ~ DDIM=20 with no
goal-reaching collapse, the ~7 s/replan cheap regime is green-lit for 6b. Pivot/pure-rotation control is
what softens first at DDIM=3, so motion type is STRATIFIED and reported per bucket, not just averaged.

Reuses UNCHANGED: CEMPlanner (action_dim=2), DiffusionWorldModel.rollout/encode_obs,
create_objective_fn(mode="last", visual_metric="mse"), the lekiwi val split (frames + GT integrate_se2
(Dx,Dtheta)). Goal sampling stratifies by motion bucket + spans episodes (cap per episode), rejects
near-trivial goals.

    python src/sample/offline_planning_eval.py \
      --ckpt /workspace/results/.../epoch=13-step=8000.ckpt \
      --out results/offline_planning_step8000 \
      --n_evals 36 --goal_H 3 --horizon 3 \
      --ddim 20 5 3 --num_samples 32 --opt_steps 3 --seed 42
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])  # put src/ on the path

from action_diagnostic import load_checkpoint                 # noqa: E402
from planning import DiffusionWorldModel                       # noqa: E402
from planning.cem_planner import CEMPlanner                    # noqa: E402
from planning.objective import create_objective_fn            # noqa: E402
from wm_datasets import create_train_val_datasets              # noqa: E402
from sampling_utils import decode_latents                      # noqa: E402

BUCKETS = ["translation", "pivot", "arc", "slow"]
RAD2DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Motion classification (over the goal_H GT chunks), in physical units.
# ---------------------------------------------------------------------------
def classify(net_dx_cm, net_dth_deg, args):
    """Bucket a goal by its net integrated GT motion. Returns a bucket name or None (trivial)."""
    adx, adth = abs(net_dx_cm), abs(net_dth_deg)
    if adx < args.trivial_dx_cm and adth < args.trivial_dth_deg:
        return None  # near-stationary -> reject (would inflate apparent success without testing planning)
    if adx >= args.arc_dx_min and adth >= args.arc_dth_min:
        return "arc"
    if adth < args.trans_dth_max and adx >= args.trivial_dx_cm:
        return "translation"
    if adx < args.rot_dx_max and adth >= args.trivial_dth_deg:
        return "pivot"
    return "slow"  # non-trivial but not cleanly trans/pivot/arc (mild combined motion)


def flat_l2(a, b):
    """Latent L2 norm between two [.., D] tensors, reduced to a python float (batch size 1)."""
    return float(torch.norm((a - b).reshape(-1)))


# ---------------------------------------------------------------------------
# Scene selection: scan val, classify, stratify by bucket, span episodes.
# ---------------------------------------------------------------------------
def select_scenes(val, goal_H, args):
    a_mean = torch.tensor(np.asarray(val._raw_action_mean).reshape(-1)[:2], dtype=torch.float32)
    a_std = torch.tensor(np.asarray(val._raw_action_std).reshape(-1)[:2], dtype=torch.float32)

    stride = max(1, len(val) // args.scan_max)
    scan_idx = list(range(0, len(val), stride))
    sub = torch.utils.data.Subset(val, scan_idx)
    loader = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=False, num_workers=4)

    recs = []  # dicts with val_idx, episode, offset, bucket, net motion, per-step lists
    k = 0
    for batch in loader:
        act = batch["action"][:, :goal_H].float()      # [B, goal_H, 2] normalized
        raw = act * a_std + a_mean                      # -> (m, rad)
        dx_cm = raw[..., 0] * 100.0
        dth_deg = raw[..., 1] * RAD2DEG
        net_dx = dx_cm.sum(1)
        net_dth = dth_deg.sum(1)
        for j in range(act.shape[0]):
            vidx = scan_idx[k + j]
            sl = val.all_slices[val.slice_indices[vidx]]
            ndx, ndth = float(net_dx[j]), float(net_dth[j])
            bucket = classify(ndx, ndth, args)
            if bucket is None:
                continue
            recs.append(dict(
                val_idx=vidx, episode=int(sl.traj_idx), offset=int(sl.start_frame),
                bucket=bucket, net_dx_cm=ndx, net_dth_deg=ndth,
                dxs=[float(v) for v in dx_cm[j]], dths=[float(v) for v in dth_deg[j]],
                mag=abs(ndx) + abs(ndth),
            ))
        k += act.shape[0]

    # Per bucket: sort by motion magnitude desc, greedily fill quota respecting the per-episode cap.
    quota = max(1, round(args.n_evals / len(BUCKETS)))
    by_bucket = defaultdict(list)
    for r in recs:
        by_bucket[r["bucket"]].append(r)

    selected, shortfalls = [], {}
    for b in BUCKETS:
        cand = sorted(by_bucket[b], key=lambda r: r["mag"], reverse=True)
        ep_count = defaultdict(int)
        picked = []
        for r in cand:
            if len(picked) >= quota:
                break
            if ep_count[r["episode"]] >= args.per_episode_cap:
                continue
            ep_count[r["episode"]] += 1
            picked.append(r)
        selected.extend(picked)
        if len(picked) < quota:
            shortfalls[b] = {"got": len(picked), "wanted": quota, "available": len(cand)}

    # Stable scene ids: bucket-major, magnitude order within bucket.
    selected.sort(key=lambda r: (BUCKETS.index(r["bucket"]), -r["mag"]))
    for i, r in enumerate(selected):
        r["scene_id"] = i
    return selected, shortfalls, (a_mean, a_std)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/offline_planning_step8000")
    ap.add_argument("--n_evals", type=int, default=36, help="target total scenes (~split evenly across 4 buckets)")
    ap.add_argument("--goal_H", type=int, default=3, help="chunks ahead the goal frame sits")
    ap.add_argument("--horizon", type=int, default=None, help="CEM planning horizon (defaults to goal_H; must equal it)")
    ap.add_argument("--ddim", type=int, nargs="+", default=[20, 5, 3], help="DDIM steps to sweep")
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--opt_steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--var_scale", type=float, default=1.0)
    ap.add_argument("--action_clip", type=float, default=None,
                    help="optional symmetric clip on normalized CEM actions (default: unclipped)")
    ap.add_argument("--seed", type=int, default=42)
    # montage
    ap.add_argument("--montage_n", type=int, default=8, help="how many scenes to render decoded montages for")
    ap.add_argument("--montage_ddim", type=int, default=50, help="DDIM steps for the clean montage rollout")
    ap.add_argument("--montage_plan_ddim", type=int, default=3,
                    help="which swept-DDIM's CEM plan to visualize (the cheap regime under test)")
    ap.add_argument("--scan_max", type=int, default=1500, help="how many strided val slices to scan")
    ap.add_argument("--per_episode_cap", type=int, default=2)
    # bucket thresholds (physical units; tuned for f=10, bang-bang LeKiwi data)
    ap.add_argument("--trivial_dx_cm", type=float, default=1.5, help="below this |Dx| AND |Dtheta| -> trivial goal, rejected")
    ap.add_argument("--trivial_dth_deg", type=float, default=4.0)
    ap.add_argument("--trans_dth_max", type=float, default=5.0, help="max |net Dtheta| for a translation goal")
    ap.add_argument("--rot_dx_max", type=float, default=2.5, help="max |net Dx| for a pivot goal")
    ap.add_argument("--arc_dx_min", type=float, default=3.0)
    ap.add_argument("--arc_dth_min", type=float, default=6.0)
    args = ap.parse_args()

    horizon = args.goal_H if args.horizon is None else args.horizon
    if horizon != args.goal_H:
        raise SystemExit(
            f"offline grading compares the rollout's LAST frame to the goal at goal_H={args.goal_H}, "
            f"so --horizon must equal --goal_H (got horizon={horizon}). "
            f"Use a longer goal_H for longer plans.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "montages").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    model, latent_codec, diffusion, train_cfg = load_checkpoint(args.ckpt, device)
    wm_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    OmegaConf.set_struct(wm_cfg, False)  # allow us to flip model.num_sampling_steps per DDIM
    world_model = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)
    vae = latent_codec.vae
    vae_precision = getattr(latent_codec, "precision", "fp32")

    H_train = train_cfg.model.num_frames - train_cfg.model.n_context_frames
    if args.goal_H > H_train:
        print(f"[note] goal_H={args.goal_H} > training horizon {H_train}: the WM rolls out "
              f"autoregressively past its train window (compounding error grows — see long_rollout_viz).")
    f = train_cfg.dataset.frame_interval

    # ---- val dataset built with num_frames = goal_H + 1 (context + goal_H chunks) ----
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    loader_cfg["validation_fixed_subset_size"] = None
    loader_cfg["validation_fixed_subset_path"] = None
    loader_cfg["validation_size"] = None
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=args.goal_H + 1,
        frame_interval=f,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )

    selected, shortfalls, (a_mean, a_std) = select_scenes(val, args.goal_H, args)
    a_mean_d, a_std_d = a_mean.to(device), a_std.to(device)
    if not selected:
        raise SystemExit("No non-trivial scenes found; loosen thresholds or raise --scan_max.")
    counts = defaultdict(int)
    for r in selected:
        counts[r["bucket"]] += 1
    print(f"Selected {len(selected)} scenes across {len({r['episode'] for r in selected})} episodes: "
          + ", ".join(f"{b}={counts[b]}" for b in BUCKETS))
    if shortfalls:
        for b, s in shortfalls.items():
            print(f"  [shortfall] bucket '{b}': got {s['got']}/{s['wanted']} (only {s['available']} available in val)")

    # ---- objective + CEM (reused unchanged) ----
    objective_fn = create_objective_fn(alpha=1.0, base=2.0, mode="last", visual_metric="mse")

    def make_planner():
        return CEMPlanner(
            world_model=world_model, objective_fn=objective_fn,
            action_dim=2, horizon=horizon,
            num_samples=args.num_samples, topk=args.topk, opt_steps=args.opt_steps,
            var_scale=args.var_scale, eval_every=args.opt_steps,  # only print final iter
            sigma_min=1e-3,
            action_low=(-args.action_clip if args.action_clip is not None else None),
            action_high=(args.action_clip if args.action_clip is not None else None),
            name="CEM", device=str(device),
        )

    def set_ddim(n):
        wm_cfg.model.num_sampling_steps = int(n)

    def rollout_last_latent(video_item, act_norm, ddim):
        """Roll the WM forward under `act_norm` [1, horizon, 2] at `ddim` steps; return flat last latent."""
        obs_0 = {"visual": video_item[:1].unsqueeze(0).to(device)}  # [1,1,C,H,W]
        z, _ = world_model.rollout(obs_0, act_norm.to(device), num_sampling_steps=ddim)
        return z["visual"][:, -1:]  # [1,1,D]

    # ---- run the DDIM sweep ----
    montage_ids = set(r["scene_id"] for r in
                      _spread_for_montage(selected, args.montage_n))
    montage_cache = {}  # scene_id -> CEM action plan at montage_plan_ddim (normalized)

    scene_rows = []
    for r in selected:
        item = val[r["val_idx"]]
        video = item["video"]                       # [1+goal_H, C, H, W]
        action = item["action"].float()             # [>=goal_H, 2] normalized
        act_gt = action[:args.goal_H].unsqueeze(0)  # [1, goal_H, 2]

        obs_0 = {"visual": video[:1].unsqueeze(0).to(device)}                    # [1,1,C,H,W]
        obs_g = {"visual": video[args.goal_H:args.goal_H + 1].unsqueeze(0).to(device)}
        z0 = world_model.encode_obs(obs_0)["visual"][:, -1:]                     # [1,1,D]
        zg = world_model.encode_obs(obs_g)["visual"][:, -1:]
        do_nothing = flat_l2(z0, zg)

        # GT raw commands (cm/deg) for the answer key
        gt_raw = (act_gt.to(device) * a_std_d + a_mean_d)[0]                     # [goal_H, 2] (m, rad)
        gt_dx_cm = gt_raw[:, 0] * 100.0
        gt_dth_deg = gt_raw[:, 1] * RAD2DEG

        row = dict(scene_id=r["scene_id"], val_idx=r["val_idx"], episode=r["episode"],
                   offset=r["offset"], bucket=r["bucket"],
                   net_dx_cm=r["net_dx_cm"], net_dth_deg=r["net_dth_deg"],
                   do_nothing=do_nothing, per_ddim={})

        for ddim in args.ddim:
            set_ddim(ddim)
            gt_ceiling = flat_l2(rollout_last_latent(video, act_gt, ddim), zg)

            planner = make_planner()
            print(f"  scene {r['scene_id']:2d} [{r['bucket']:11s}] DDIM={ddim:2d} "
                  f"net Dx={r['net_dx_cm']:+5.1f}cm Dtheta={r['net_dth_deg']:+5.1f}°", flush=True)
            mu, info = planner.plan(obs_0, obs_g)        # mu: [1, horizon, 2] normalized
            mu = mu.detach()
            cem_reached = flat_l2(rollout_last_latent(video, mu, ddim), zg)
            print(f"      do_nothing={do_nothing:6.2f}  gt_ceiling={gt_ceiling:6.2f}  "
                  f"cem_reached={cem_reached:6.2f}  "
                  f"{'BEATS' if cem_reached < do_nothing else 'flat '}  "
                  f"(wm_drop={do_nothing - gt_ceiling:+5.2f}, cem_drop={do_nothing - cem_reached:+5.2f})",
                  flush=True)

            # action recovery (denormalize CEM plan -> raw (Dx,Dtheta))
            cem_raw = (mu.to(device) * a_std_d + a_mean_d)[0]                    # [horizon, 2]
            cem_dx_cm = cem_raw[:, 0] * 100.0
            cem_dth_deg = cem_raw[:, 1] * RAD2DEG
            recov_norm = float(torch.norm(mu.to(device)[0] - act_gt.to(device)[0], dim=-1).mean())
            net_cem_dx = float(cem_dx_cm.sum()); net_cem_dth = float(cem_dth_deg.sum())
            net_gt_dx = float(gt_dx_cm.sum()); net_gt_dth = float(gt_dth_deg.sum())

            row["per_ddim"][str(ddim)] = dict(
                gt_ceiling=gt_ceiling, cem_reached=cem_reached,
                beats_do_nothing=bool(cem_reached < do_nothing),
                reached_ratio=float(cem_reached / gt_ceiling) if gt_ceiling > 0 else None,
                cem_minus_donothing=float(cem_reached - do_nothing),
                action_recovery_norm=recov_norm,
                dx_err_cm=float((cem_dx_cm - gt_dx_cm).abs().mean()),
                dth_err_deg=float((cem_dth_deg - gt_dth_deg).abs().mean()),
                net_cem_dx_cm=net_cem_dx, net_gt_dx_cm=net_gt_dx,
                net_cem_dth_deg=net_cem_dth, net_gt_dth_deg=net_gt_dth,
                # sign match is only meaningful when the GT component is non-negligible
                # (a pivot goal has net Dx~0, so its Dx sign is noise -> report None, excluded from the rate).
                dx_sign_match=(None if abs(net_gt_dx) < args.trivial_dx_cm
                               else bool(np.sign(net_cem_dx) == np.sign(net_gt_dx))),
                dth_sign_match=(None if abs(net_gt_dth) < args.trivial_dth_deg
                                else bool(np.sign(net_cem_dth) == np.sign(net_gt_dth))),
                cem_final_loss=float(info["final_loss"]),
            )
            if r["scene_id"] in montage_ids and ddim == args.montage_plan_ddim:
                montage_cache[r["scene_id"]] = mu.cpu()

        scene_rows.append(row)

    # ---- montages (decoded) for a spread of scenes/buckets ----
    print(f"\nRendering {len(montage_cache)} decoded montages (plan DDIM={args.montage_plan_ddim}, "
          f"rollout DDIM={args.montage_ddim})...")
    ref = latent_codec.encode(val[selected[0]['val_idx']]["video"][:1].to(device))
    C_lat, hl, wl = ref.shape[1:]
    for r in selected:
        sid = r["scene_id"]
        if sid not in montage_cache:
            continue
        item = val[r["val_idx"]]
        video = item["video"].to(device)
        mu = montage_cache[sid].to(device)
        set_ddim(args.montage_ddim)
        z_cem, _ = world_model.rollout({"visual": video[:1].unsqueeze(0)}, mu,
                                       num_sampling_steps=args.montage_ddim)
        cem_lat = z_cem["visual"].reshape(1, 1 + args.goal_H, C_lat, hl, wl)
        gt_lat = torch.stack([latent_codec.encode(video[t:t + 1]) for t in range(1 + args.goal_H)], dim=1)
        cem_frames = decode_latents(vae, cem_lat, vae_precision).clamp(0, 1).cpu().numpy()[0]
        gt_frames = decode_latents(vae, gt_lat, vae_precision).clamp(0, 1).cpu().numpy()[0]
        # latentL2 of CEM-planned final vs goal
        m = r_metrics(scene_rows, sid, args.montage_plan_ddim)
        _save_montage(out / "montages" / f"scene{sid:02d}_{r['bucket']}.png",
                      gt_frames, cem_frames, args.goal_H, r, m, args.montage_plan_ddim)

    # ---- aggregate (overall + per bucket, per DDIM) ----
    summary = aggregate(scene_rows, args.ddim)

    result = dict(
        meta=dict(
            ckpt=args.ckpt, n_evals_target=args.n_evals, n_scenes=len(scene_rows),
            goal_H=args.goal_H, horizon=horizon, frame_interval=f, train_horizon=H_train,
            ddim_sweep=args.ddim, num_samples=args.num_samples, opt_steps=args.opt_steps,
            topk=args.topk, var_scale=args.var_scale, action_clip=args.action_clip, seed=args.seed,
            action_mean=a_mean.tolist(), action_std=a_std.tolist(),
            bucket_counts={b: counts[b] for b in BUCKETS}, shortfalls=shortfalls,
            n_episodes=len({r["episode"] for r in scene_rows}),
        ),
        scenes=scene_rows,
        summary=summary,
    )
    with open(out / "offline_planning_eval.json", "w") as fjson:
        json.dump(result, fjson, indent=2)

    print_report(summary, args.ddim, out)


def _spread_for_montage(selected, n):
    """Pick ~n scenes spread across buckets for montage rendering."""
    if n <= 0:
        return []
    by_bucket = defaultdict(list)
    for r in selected:
        by_bucket[r["bucket"]].append(r)
    picked, bi = [], 0
    order = [b for b in BUCKETS if by_bucket[b]]
    cursors = {b: 0 for b in order}
    while len(picked) < min(n, len(selected)) and order:
        b = order[bi % len(order)]
        if cursors[b] < len(by_bucket[b]):
            picked.append(by_bucket[b][cursors[b]])
            cursors[b] += 1
        bi += 1
        if all(cursors[b] >= len(by_bucket[b]) for b in order):
            break
    return picked


def r_metrics(scene_rows, sid, ddim):
    for row in scene_rows:
        if row["scene_id"] == sid:
            return row["per_ddim"].get(str(ddim), {}) | {"do_nothing": row["do_nothing"]}
    return {}


def _save_montage(path, gt_frames, cem_frames, goal_H, r, m, plan_ddim):
    titles = ["context"] + [f"t+{i}" for i in range(1, 1 + goal_H)]
    fig, axes = plt.subplots(2, 1 + goal_H, figsize=(3 * (1 + goal_H), 6.2))
    for t in range(1 + goal_H):
        axes[0, t].imshow(np.transpose(gt_frames[t], (1, 2, 0)))
        axes[0, t].set_title(("GOAL " if t == goal_H else "GT ") + titles[t],
                             fontsize=9, color=("C2" if t == goal_H else "k"))
        axes[0, t].axis("off")
        axes[1, t].imshow(np.transpose(cem_frames[t], (1, 2, 0)))
        axes[1, t].set_title(("CEM-planned " if t > 0 else "start ") + titles[t], fontsize=9)
        axes[1, t].axis("off")
    dn, gc, cr = m.get("do_nothing"), m.get("gt_ceiling"), m.get("cem_reached")
    fig.suptitle(
        f"{r['bucket'].upper()} scene {r['scene_id']} (ep{r['episode']} off{r['offset']}) — "
        f"net Dx={r['net_dx_cm']:+.1f}cm Dtheta={r['net_dth_deg']:+.1f}°\n"
        f"top: real GT path (last=GOAL)   bottom: WM rollout under CEM plan (DDIM={plan_ddim})   |   "
        f"do_nothing={dn:.1f}  gt_ceiling={gc:.1f}  cem_reached={cr:.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def aggregate(scene_rows, ddim_list):
    summary = {}
    for ddim in ddim_list:
        d = str(ddim)
        per_bucket = {}
        for b in BUCKETS + ["overall"]:
            rows = [r for r in scene_rows if (b == "overall" or r["bucket"] == b)]
            rows = [r for r in rows if d in r["per_ddim"]]
            if not rows:
                continue

            def bmean(key):
                vals = [r["per_ddim"][d][key] for r in rows if r["per_ddim"][d].get(key) is not None]
                return float(np.mean(vals)) if vals else None

            per_bucket[b] = dict(
                n=len(rows),
                do_nothing=float(np.mean([r["do_nothing"] for r in rows])),
                gt_ceiling=bmean("gt_ceiling"),
                cem_reached=bmean("cem_reached"),
                reached_ratio=bmean("reached_ratio"),
                beats_do_nothing_rate=float(np.mean([r["per_ddim"][d]["beats_do_nothing"] for r in rows])),
                action_recovery_norm=bmean("action_recovery_norm"),
                dx_err_cm=bmean("dx_err_cm"),
                dth_err_deg=bmean("dth_err_deg"),
                dx_sign_match_rate=bmean("dx_sign_match"),
                dth_sign_match_rate=bmean("dth_sign_match"),
            )
        summary[d] = per_bucket
    return summary


def print_report(summary, ddim_list, out):
    print("\n" + "=" * 96)
    print("OFFLINE PLANNING EVAL — summary (latent-L2; lower cem_reached is better; reached_ratio→1 = WM-optimal)")
    print("=" * 96)
    for ddim in ddim_list:
        d = str(ddim)
        print(f"\nDDIM={ddim}")
        print(f"  {'bucket':12s} {'n':>3s} {'do_noth':>8s} {'gt_ceil':>8s} {'cem_rch':>8s} "
              f"{'ratio':>6s} {'beat%':>6s} {'recov':>6s} {'dxErr':>6s} {'dthErr':>7s} {'dxSgn':>6s} {'dthSgn':>6s}")
        for b in ["translation", "pivot", "arc", "slow", "overall"]:
            s = summary[d].get(b)
            if not s:
                continue
            print(f"  {b:12s} {s['n']:3d} {s['do_nothing']:8.1f} {s['gt_ceiling']:8.1f} {s['cem_reached']:8.1f} "
                  f"{(s['reached_ratio'] or 0):6.2f} {s['beats_do_nothing_rate']*100:5.0f}% "
                  f"{(s['action_recovery_norm'] or 0):6.2f} {(s['dx_err_cm'] or 0):6.1f} {(s['dth_err_deg'] or 0):7.1f} "
                  f"{(s['dx_sign_match_rate'] or 0)*100:5.0f}% {(s['dth_sign_match_rate'] or 0)*100:5.0f}%")

    # cheap-sampler hold: pivot is what softens first
    if len(ddim_list) > 1:
        hi, lo = str(max(ddim_list)), str(min(ddim_list))
        print(f"\nCheap-sampler hold (DDIM={lo} vs DDIM={hi}) — cem_reached delta per bucket "
              f"(positive = worse at the cheap setting):")
        for b in ["translation", "pivot", "arc", "slow", "overall"]:
            shi, slo = summary[hi].get(b), summary[lo].get(b)
            if shi and slo:
                dlt = slo["cem_reached"] - shi["cem_reached"]
                flag = "  <-- softens first (watch)" if b == "pivot" else ""
                print(f"  {b:12s} {shi['cem_reached']:7.1f} -> {slo['cem_reached']:7.1f}  (Δ {dlt:+5.1f}){flag}")
    print(f"\nWrote {out/'offline_planning_eval.json'} + montages/ ({out/'montages'})")


if __name__ == "__main__":
    main()
