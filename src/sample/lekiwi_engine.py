"""
Stage 6b.2 — LeKiwi live planning engine (NanoNAV).

Wraps the EXACT 6a-validated path (offline_planning_eval.py) behind a live-frame interface for the
closed-loop controller (NanoNAV scripts/lekiwi_mpc.py, --planner wm). Same load + DiffusionWorldModel
+ CEMPlanner + integrate_se2 action stats + latent-L2 "last" objective. The only new pieces are:
  (1) preprocessing a raw robot `top` frame to match the dataset's training transform, and
  (2) returning a PlanResult (first-chunk velocity + current latent-dist-to-goal + decoded imagined
      rollout and top-K elite fan for rerun).

Runs on the POD (GPU + checkpoint + dataset). Authored to mirror 6a call-for-call; VERIFY on the pod:
  • action stats print should read mean~[0.0221,-0.0006], std~[0.0141,0.0707] (the integrate_se2 f=10 stats);
  • a do_nothing sanity (frame vs itself) should give dist≈0;
  • decoded `imagined` should look like a plausible top-view, not noise (confirms the [0,1] frame range).
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Make the fork's src/ and src/sample/ importable regardless of how we're launched.
_HERE = os.path.dirname(os.path.abspath(__file__))     # .../src/sample
_SRC = os.path.dirname(_HERE)                           # .../src
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from action_diagnostic import load_checkpoint            # noqa: E402  (src/sample)
from planning import DiffusionWorldModel                 # noqa: E402
from planning.cem_planner import CEMPlanner              # noqa: E402
from planning.objective import create_objective_fn       # noqa: E402
from wm_datasets import create_train_val_datasets         # noqa: E402
from sampling_utils import decode_latents                # noqa: E402

CHUNK_DT = 10.0 / 30.0                                    # f·Δt = 0.333 s (Run-002 chunk)
RAD2DEG = 180.0 / math.pi


@dataclass
class PlanResult:
    vx: float                                # m/s, first chunk
    theta_deg: float                         # deg/s, first chunk (+ = CCW, per 6b.0)
    dist_to_goal: float                      # current latent-L2(z_now, z_goal) — termination signal
    cem_loss: Optional[float] = None
    imagined_rgb: Optional[np.ndarray] = None
    elite_rgb: List[np.ndarray] = field(default_factory=list)


def _flat_l2(a, b):
    return float(torch.norm((a - b).reshape(-1)))


class LekiwiPlanner:
    """Live wrapper of the 6a planner. plan(frame, goal) -> PlanResult (matches scripts/lekiwi_mpc.py)."""

    def __init__(self, ckpt, device="cuda", ddim=3, num_samples=32, opt_steps=3, topk=10,
                 horizon=3, n_elite_viz=3, action_mean=None, action_std=None):
        self.device = torch.device(device)
        self.ddim, self.horizon, self.n_elite_viz = int(ddim), int(horizon), int(n_elite_viz)
        self.num_samples, self.opt_steps, self.topk = int(num_samples), int(opt_steps), int(topk)
        torch.set_grad_enabled(False)

        # ---- model (exactly as 6a) ----
        model, latent_codec, diffusion, train_cfg = load_checkpoint(ckpt, self.device)
        wm_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
        OmegaConf.set_struct(wm_cfg, False)              # allow flipping model.num_sampling_steps per DDIM
        self.wm = DiffusionWorldModel(model, latent_codec, diffusion, wm_cfg)
        self.wm_cfg = wm_cfg
        self.latent_codec = latent_codec
        self.vae = latent_codec.vae
        self.vae_precision = getattr(latent_codec, "precision", "fp32")
        self.f = train_cfg.dataset.frame_interval
        img = train_cfg.model.image_size
        self.image_size = (img, img) if isinstance(img, int) else tuple(img)

        H_train = train_cfg.model.num_frames - train_cfg.model.n_context_frames
        if self.horizon > H_train:
            print(f"[engine] horizon {self.horizon} > train horizon {H_train}: WM rolls out autoregressively "
                  f"past its train window (compounding error grows).")

        # ---- integrate_se2 action stats for denorm(CEM) -> (Δx m, Δθ rad), exactly like 6a ----
        if action_mean is not None and action_std is not None:
            a_mean = torch.tensor(action_mean, dtype=torch.float32)
            a_std = torch.tensor(action_std, dtype=torch.float32)
        else:
            loader_cfg = OmegaConf.to_container(train_cfg.dataset.loader, resolve=True)
            loader_cfg["validation_fixed_subset_size"] = None
            loader_cfg["validation_fixed_subset_path"] = None
            loader_cfg["validation_size"] = None
            _, val = create_train_val_datasets(
                dataset_name=train_cfg.dataset.name, num_frames=2, frame_interval=self.f,
                image_size=train_cfg.model.image_size, **loader_cfg)
            a_mean = torch.tensor(np.asarray(val._raw_action_mean).reshape(-1)[:2], dtype=torch.float32)
            a_std = torch.tensor(np.asarray(val._raw_action_std).reshape(-1)[:2], dtype=torch.float32)
        self.a_mean, self.a_std = a_mean.to(self.device), a_std.to(self.device)
        print(f"[engine] action stats mean={a_mean.tolist()} std={a_std.tolist()} "
              f"(expect ~[0.0221,-0.0006] / [0.0141,0.0707])")

        # latent shape for decode reshape (SD-VAE → [C_lat,h,w])
        ref = self.latent_codec.encode(torch.zeros(1, 3, *self.image_size, device=self.device))
        self.C_lat, self.h_lat, self.w_lat = ref.shape[1:]

        self.objective_fn = create_objective_fn(alpha=1.0, base=2.0, mode="last", visual_metric="mse")
        self._goal_cache = None                          # (id(goal), obs_g, zg)
        print(f"[engine] ready: ckpt loaded, f={self.f}, image_size={self.image_size}, "
              f"latent=[{self.C_lat},{self.h_lat},{self.w_lat}], DDIM={self.ddim}, "
              f"CEM {self.num_samples}×{self.opt_steps}×top{self.topk}, H={self.horizon}")

    # ---- helpers ----
    def _make_planner(self):
        return CEMPlanner(
            world_model=self.wm, objective_fn=self.objective_fn, action_dim=2, horizon=self.horizon,
            num_samples=self.num_samples, topk=self.topk, opt_steps=self.opt_steps,
            var_scale=1.0, eval_every=self.opt_steps, sigma_min=1e-3,
            action_low=None, action_high=None, name="CEM", device=str(self.device))

    def _set_ddim(self, n):
        self.wm_cfg.model.num_sampling_steps = int(n)

    def _preprocess(self, frame):
        """
        Raw HWC RGB robot frame -> [1,1,C,256,256] in [0,1], letterbox-padded — IDENTICAL to the
        dataset transform (world_model_dataset.py:614-641; lerobot loader returns [0,1] CHW RGB).
        """
        t = torch.as_tensor(np.ascontiguousarray(frame))
        if t.ndim == 3 and t.shape[2] in (1, 3):         # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.float()
        if float(t.max()) > 1.5:                          # uint8 -> [0,1]
            t = t / 255.0
        t = t.unsqueeze(0)                                # [1,C,H,W]
        target_h, target_w = self.image_size
        _, _, H, W = t.shape
        if (H, W) != (target_h, target_w):
            scale = min(target_h / H, target_w / W)
            new_h, new_w = int(H * scale), int(W * scale)
            t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            pad_h, pad_w = target_h - new_h, target_w - new_w
            pad_top, pad_left = pad_h // 2, pad_w // 2
            t = F.pad(t, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
        return t.unsqueeze(1).to(self.device)            # [1,1,C,H,W]

    def _encode_last(self, obs):
        return self.wm.encode_obs(obs)["visual"][:, -1:]

    def _decode_last(self, latents_flat):
        lat = latents_flat.reshape(1, 1, self.C_lat, self.h_lat, self.w_lat)
        img = decode_latents(self.vae, lat, self.vae_precision).clamp(0, 1)   # [1,1,C,H,W] in [0,1]
        return (img[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _goal(self, goal):
        if self._goal_cache and self._goal_cache[0] == id(goal):
            return self._goal_cache[1], self._goal_cache[2]
        obs_g = {"visual": self._preprocess(goal)}
        zg = self._encode_last(obs_g)
        self._goal_cache = (id(goal), obs_g, zg)
        return obs_g, zg

    # ---- the one method the controller calls ----
    @torch.no_grad()
    def plan(self, frame, goal) -> PlanResult:
        obs_0 = {"visual": self._preprocess(frame)}
        z0 = self._encode_last(obs_0)
        obs_g, zg = self._goal(goal)
        dist_to_goal = _flat_l2(z0, zg)                  # current latent-L2 to goal (termination + logging)

        self._set_ddim(self.ddim)
        planner = self._make_planner()
        mu, info = planner.plan(obs_0, obs_g, return_elites=(self.n_elite_viz > 0))  # mu [1,H,2] normalized

        raw = (mu.to(self.device) * self.a_std + self.a_mean)[0]   # [H,2] (m, rad)
        dx, dth = float(raw[0, 0]), float(raw[0, 1])               # FIRST chunk only (execute-one, replan)
        vx = dx / CHUNK_DT
        theta_deg = (dth / CHUNK_DT) * RAD2DEG

        # decoded imagined rollout (WM's predicted goal frame under the plan) + top-K elite fan, for rerun
        imagined_rgb, elite_rgb = None, []
        try:
            z_cem, _ = self.wm.rollout(obs_0, mu.to(self.device), num_sampling_steps=self.ddim)
            imagined_rgb = self._decode_last(z_cem["visual"][:, -1:])
            elites = info.get("elite_actions")
            if elites is not None and self.n_elite_viz > 0:
                for k in range(min(self.n_elite_viz, elites.shape[0])):
                    ze, _ = self.wm.rollout(obs_0, elites[k:k + 1].to(self.device), num_sampling_steps=self.ddim)
                    elite_rgb.append(self._decode_last(ze["visual"][:, -1:]))
        except Exception as e:
            print(f"[engine] viz decode skipped ({e})")

        return PlanResult(vx=vx, theta_deg=theta_deg, dist_to_goal=dist_to_goal,
                          cem_loss=float(info.get("final_loss", 0.0)),
                          imagined_rgb=imagined_rgb, elite_rgb=elite_rgb)
