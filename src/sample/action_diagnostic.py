"""
Table 5/6 action-conditioning diagnostic (NanoNAV Stage-5 gate).

Does the trained world model actually USE the action? Rolls a held-out context frame forward under
three action conditions and measures how close the predicted final latent lands to the true future:

  * GT      — the dataset's real (Δx, Δθ) actions
  * zero    — all-zeros actions
  * random  — Gaussian actions in normalized action space

PASS:  GT latent-L2 clearly below zero/random, and action-embedding RMS ~0.1+.
FAIL:  the three are comparable / RMS ~0.002  → the action branch atrophied (Finding #4); fix training
       before any planning (aux pose / cross-attn injection / larger embed / augmentation).
See NanoNAV context/training.md and context/runpod-operator-guide.md.

The same context frame, horizon, and latent-distance metric are used for all three conditions, so the
relative comparison is robust to the exact action/frame index alignment.

Usage (on the pod, after training):
    python src/sample/action_diagnostic.py --ckpt $RESULTS_DIR/<run>/checkpoints/latest-*.ckpt \
        --dataset lerobot/lekiwi --n-batches 8 --out $RESULTS_DIR/<run>/action_diag
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Make the repo's src/ importable when run as `python src/sample/action_diagnostic.py` from the repo
# root (mirrors rollout.py / sample_dfot.py). Without it, `from diffusion import ...` raises
# ModuleNotFoundError.
sys.path.append(os.path.split(sys.path[0])[0])

from diffusion import create_diffusion
from latent_codecs import build_latent_codec
from models import get_models
from planning import DiffusionWorldModel
from wm_datasets import create_train_val_datasets

# The saved training config carries ${hydra:...} interpolations (e.g. planning.output_dir) that only
# resolve inside a live Hydra run. Register a stub so OmegaConf.resolve() doesn't crash when we load
# that config here for inference (the diagnostic never uses those fields).
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda *a, **k: ".")


def load_checkpoint(ckpt_path: str, device):
    """Mirror PlanningExperiment._load_from_checkpoint: (model, latent_codec, diffusion, train_cfg)."""
    ckpt_path_str = str(ckpt_path)
    if "/" in ckpt_path_str and not Path(ckpt_path_str).exists():
        from huggingface_hub import hf_hub_download
        train_cfg = OmegaConf.load(hf_hub_download(ckpt_path_str, "config.yaml"))
        model = get_models(train_cfg).to(device)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(hf_hub_download(ckpt_path_str, "model.safetensors"), device=str(device)))
    else:
        ckpt_path = Path(ckpt_path_str).expanduser().resolve()
        cfg_path = next((a / "config.yaml" for a in [ckpt_path.parent] + list(ckpt_path.parents)
                         if (a / "config.yaml").exists()), None)
        if cfg_path is None:
            raise FileNotFoundError(f"config.yaml not found near {ckpt_path}")
        train_cfg = OmegaConf.load(cfg_path)
        model = get_models(train_cfg).to(device)
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            model.load_state_dict(load_file(str(ckpt_path), device=str(device)))
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            sd = {k.removeprefix("model.").removeprefix("_orig_mod."): v
                  for k, v in sd.items() if not k.removeprefix("model.").startswith("vae.")}
            model.load_state_dict(sd, strict=False)
    model.eval()
    latent_codec = build_latent_codec(train_cfg).to(device).eval()
    d = train_cfg.experiment.diffusion
    diffusion = create_diffusion(
        timestep_respacing="", noise_schedule=d.noise_schedule, diffusion_steps=d.diffusion_steps,
        pred_name=d.pred_name, snr_gamma=d.snr_gamma, zero_terminal_snr=d.zero_terminal_snr,
    )
    return model, latent_codec, diffusion, train_cfg


@torch.no_grad()
def encode_final_target(latent_codec, frame, device):
    """Encode one [B,C,H,W] frame to a flat latent [B, D] matching rollout's z_visual layout."""
    z = latent_codec.encode(frame.to(device))      # [B, C_lat, H_lat, W_lat]
    return z.reshape(z.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="lerobot/lekiwi", help="hydra dataset name (informational)")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--num-sampling-steps", type=int, default=20)
    ap.add_argument("--out", default="action_diag")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, latent_codec, diffusion, train_cfg = load_checkpoint(args.ckpt, device)
    wm_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    wm_cfg.model.num_sampling_steps = args.num_sampling_steps
    world_model = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)

    H = train_cfg.model.num_frames - train_cfg.model.n_context_frames

    # Validation set from the SAME loader config the checkpoint trained with (integrate_se2, norm).
    loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
    _, val = create_train_val_datasets(
        dataset_name=train_cfg.dataset.name,
        num_frames=train_cfg.model.num_frames,
        frame_interval=train_cfg.dataset.frame_interval,
        image_size=train_cfg.model.image_size,
        **loader_cfg,
    )
    loader = torch.utils.data.DataLoader(val, batch_size=8, shuffle=False, num_workers=2)

    # Hook the action embedder to record its output RMS (atrophy indicator).
    emb_sq, emb_n = [], []
    def hook(_m, _i, o):
        emb_sq.append(float((o.detach() ** 2).mean()) * o.numel()); emb_n.append(o.numel())
    handle = model.action_embedder.register_forward_hook(hook)

    conditions = ["gt", "zero", "random"]
    per_step = {c: [] for c in conditions}     # latent-L2 per horizon step
    final = {c: [] for c in conditions}        # latent-L2 at final step

    n = 0
    for batch in loader:
        if n >= args.n_batches:
            break
        video = batch["video"].to(device)          # [B, T, C, H, W]
        action = batch["action"].to(device)        # [B, T, 2] normalized (Δx, Δθ)
        B = video.shape[0]
        obs_0 = {"visual": video[:, :1]}            # context frame
        act_gt = action[:, :H]                      # [B, H, 2]

        # True future latents (frames 1..H)
        targets = [encode_final_target(latent_codec, video[:, t], device) for t in range(1, H + 1)]

        acts = {
            "gt": act_gt,
            "zero": torch.zeros_like(act_gt),
            "random": torch.randn_like(act_gt),     # normalized space ~ N(0,1)
        }
        for c, a in acts.items():
            z_obses, _ = world_model.rollout(obs_0, a, num_sampling_steps=args.num_sampling_steps)
            pred = z_obses["visual"]                # [B, 1+H, D]
            step_l2 = [torch.norm(pred[:, t + 1] - targets[t], dim=-1).mean().item() for t in range(H)]
            per_step[c].append(step_l2)
            final[c].append(step_l2[-1])
        n += 1

    handle.remove()
    action_rms = float(np.sqrt(sum(emb_sq) / max(sum(emb_n), 1)))

    # ---- report ----
    def mean_final(c): return float(np.mean(final[c]))
    gt_f, zero_f, rand_f = mean_final("gt"), mean_final("zero"), mean_final("random")
    passed = (gt_f < zero_f) and (gt_f < rand_f) and (action_rms > 0.05)

    print("\n=== Table 5/6 action-conditioning diagnostic ===")
    print(f"  batches: {n}   horizon: {H}   sampling steps: {args.num_sampling_steps}")
    print(f"  final-latent L2   GT={gt_f:.4f}  zero={zero_f:.4f}  random={rand_f:.4f}")
    print(f"  action-embed RMS  {action_rms:.4f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}  "
          f"(GT<{min(zero_f, rand_f):.4f}? {gt_f < min(zero_f, rand_f)}; RMS>0.05? {action_rms > 0.05})")

    # latent-L2 curves
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in conditions:
            curve = np.mean(np.array(per_step[c]), axis=0)
            ax.plot(range(1, H + 1), curve, marker="o", label=c)
        ax.set_xlabel("rollout step"); ax.set_ylabel("latent L2 to true future")
        ax.set_title(f"Action conditioning (RMS={action_rms:.3f}, {'PASS' if passed else 'FAIL'})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "action_diagnostic.png", dpi=120)
        print(f"  figure -> {out/'action_diagnostic.png'}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    summary = dict(batches=n, horizon=H, gt=gt_f, zero=zero_f, random=rand_f,
                   action_rms=action_rms, passed=bool(passed))
    (out / "action_diagnostic.json").write_text(__import__("json").dumps(summary, indent=2))
    print(f"  summary -> {out/'action_diagnostic.json'}")


if __name__ == "__main__":
    main()
