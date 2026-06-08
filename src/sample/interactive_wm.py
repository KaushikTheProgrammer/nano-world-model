"""
Interactive world-model "driving" evaluator (NanoNAV).

Drive the LeKiwi around *inside the world model's imagination* with your keyboard and watch it
generate the resulting camera frames. This is the OPEN-LOOP complement to Stage 6a's CEM eval: there,
a goal is given and the planner searches for the action; here, YOU supply the action and the WM predicts
the frame. Driving manually builds intuition for where the WM's dynamics are faithful and where they
drift / compound error — exactly the model CEM trusts when it scores imagined rollouts. At any point you
can press `c` to ask "from the frame I've driven to now, what would CEM do toward the loaded goal?" and
see its full imagined trajectory next to yours.

The pod is headless, so this serves a tiny single-page web app (stdlib http.server, no Flask): the
browser captures WASD/arrow keydowns and renders the decoded frames; the server steps the WM. Reach it
over the SSH reverse tunnel / RunPod TCP port (the scripts/tunnel_up.sh pattern), then open
http://localhost:<port>.

Reuses the 6a/6b-validated machinery via LekiwiPlanner (loads the ckpt once; exposes wm / vae / action
stats / CEM factory). The one deliberate divergence from LekiwiPlanner._preprocess: we encode frames in
the TRAINING pixel range (normalize_pixel from the ckpt config — [-1,1] for Run 002), matching the
validated offline_planning_eval path, instead of the engine's [0,1].

Open-loop semantics: we keep the start frame FIXED and re-roll wm.rollout(start, full_action_list) every
keypress (deterministic at eta=0, so earlier frames reproduce identically and the strip is stable; this
is exactly how CEM rolls out, and it surfaces the true compounding error). Only the new last frame is
decoded per step.

    export LEKIWI_DATA_ROOT=/workspace/data/lekiwi
    python src/sample/interactive_wm.py --start-val 12 --goal-val 15 --port 8765
    # or seed from images:
    python src/sample/interactive_wm.py --start-image start.png --goal-image goal.png --port 8765

Keys (in the browser): w/s forward/back, a/d turn left/right, q/e forward-arc left/right,
[ / ] shrink/grow the step size, c run CEM overlay, g set goal = current frame, u undo, r reset.
"""
import os
import sys
import io
import json
import math
import base64
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.append(os.path.split(sys.path[0])[0])  # put src/ on the path (mirrors offline_planning_eval)

from lekiwi_engine import LekiwiPlanner, CHUNK_DT, RAD2DEG       # noqa: E402  (src/sample)
from wm_datasets import create_train_val_datasets                # noqa: E402
from sampling_utils import decode_latents                        # noqa: E402

# integrate_se2 (Δx m, Δθ rad) f=10 stats — defaults, also wired in scripts/lekiwi_mpc.py.
INTEGRATE_SE2_ACTION_MEAN = [0.022110389545559883, -0.0005879045929759741]
INTEGRATE_SE2_ACTION_STD = [0.014105414971709251, 0.07071184366941452]

# key -> (forward sign, turn sign) in units of the current step magnitudes.
KEY_ACTIONS = {
    "w": (+1.0, 0.0), "s": (-1.0, 0.0),
    "a": (0.0, +1.0), "d": (0.0, -1.0),     # +Δθ = CCW / left (per 6b.0)
    "q": (+1.0, +1.0), "e": (+1.0, -1.0),   # forward-arcs
}


def _flat_l2(a, b):
    return float(torch.norm((a - b).reshape(-1)))


def load_image_rgb(path):
    """Any image file -> HWC uint8 RGB (the capture_goal.py reader, inlined to avoid a scripts/ import)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


def to_uint8(frame_chw_or_hwc, model_range):
    """A single frame tensor/array -> HWC uint8 for display. model_range=True means it's in [-1,1]."""
    t = torch.as_tensor(np.asarray(frame_chw_or_hwc)) if not torch.is_tensor(frame_chw_or_hwc) else frame_chw_or_hwc
    t = t.detach().float().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3):     # CHW -> HWC
        t = t.permute(1, 2, 0)
    if model_range:
        t = (t + 1.0) / 2.0
    elif float(t.max()) > 1.5:
        t = t / 255.0
    return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)


def png_b64(hwc_uint8):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(hwc_uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Driver:
    """Open-loop WM stepping state + the CEM overlay, around a single LekiwiPlanner."""

    def __init__(self, args):
        self.lp = LekiwiPlanner(
            ckpt=args.ckpt, device=args.device, ddim=args.ddim,
            num_samples=args.num_samples, opt_steps=args.opt_steps, topk=args.topk,
            horizon=args.horizon, n_elite_viz=args.n_elite_viz,
            action_mean=args.action_mean, action_std=args.action_std,
        )
        self.lp._set_ddim(self.lp.ddim)   # default the config to the cheap DDIM for any path that reads it
        self.device = self.lp.device
        self.base_dx = float(args.step_dx)                      # m per keystroke (forward)
        self.base_dth = math.radians(float(args.step_dth_deg))  # rad per keystroke (turn)
        self.scale = 1.0
        # training pixel range (so encoding matches the validated 6a path, not the engine's [0,1])
        loader = OmegaConf.to_container(self.lp.wm_cfg.dataset.loader, resolve=True)
        self.normalize_pixel = bool(loader.get("normalize_pixel", True))
        self.image_size = self.lp.image_size
        self._val = None      # val dataset, built lazily on first val-seed

        # --- start frame ---
        self.start_model = self._seed_model(args.start_val, args.start_image)   # [1,1,C,H,W]
        self.start_rgb = to_uint8(self.start_model[0, 0], model_range=self.normalize_pixel)
        self.start_val = args.start_val if args.start_image is None else None    # current val idx (None if image)
        self.n_val = len(self._build_val())                                      # total val slices, for the switcher

        # --- goal (optional) ---
        self.goal_model = None
        self.goal_rgb = None
        self.zg = None
        if args.goal_val is not None or args.goal_image is not None:
            self.set_goal(self._seed_model(args.goal_val, args.goal_image))

        # rolling state
        self.acts = []        # list of normalized action tensors [2] on device
        self.raw = []         # list of (dx_m, dth_rad) physical, for the HUD
        self.strip = [self.start_rgb]   # decoded frames, index 0 == start
        self.lock = threading.Lock()

    # ---- seeding ----
    def _build_val(self):
        if self._val is None:
            tc = self.lp.wm_cfg
            loader_cfg = OmegaConf.to_container(tc.dataset.loader, resolve=True)
            loader_cfg["validation_fixed_subset_size"] = None
            loader_cfg["validation_fixed_subset_path"] = None
            loader_cfg["validation_size"] = None
            _, self._val = create_train_val_datasets(
                dataset_name=tc.dataset.name, num_frames=2, frame_interval=self.lp.f,
                image_size=tc.model.image_size, **loader_cfg)
        return self._val

    def _to_model(self, rgb_uint8):
        """HWC uint8 RGB -> [1,1,C,H,W] in the TRAINING pixel range (letterbox + optional *2-1)."""
        t = torch.as_tensor(np.ascontiguousarray(rgb_uint8)).float()
        if t.ndim == 3 and t.shape[2] in (1, 3):
            t = t.permute(2, 0, 1)               # HWC -> CHW
        if float(t.max()) > 1.5:
            t = t / 255.0                        # -> [0,1]
        t = t.unsqueeze(0)                       # [1,C,H,W]
        th, tw = self.image_size
        _, _, H, W = t.shape
        if (H, W) != (th, tw):                   # letterbox (resize_mode="pad")
            scale = min(th / H, tw / W)
            nh, nw = int(H * scale), int(W * scale)
            t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
            ph, pw = th - nh, tw - nw
            t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)
        if self.normalize_pixel:
            t = t * 2.0 - 1.0                    # [0,1] -> [-1,1], exactly like the dataset
        return t.unsqueeze(1).to(self.device)    # [1,1,C,H,W]

    def _seed_model(self, val_idx, img_path):
        if img_path is not None:
            return self._to_model(load_image_rgb(img_path))
        if val_idx is not None:
            val = self._build_val()
            if not (0 <= val_idx < len(val)):
                raise SystemExit(f"--*-val {val_idx} out of range (val has {len(val)} slices)")
            return val[val_idx]["video"][:1].unsqueeze(0).to(self.device)  # dataset-native [-1,1]
        raise SystemExit("need a start/goal source (--start-val/--start-image, --goal-val/--goal-image)")

    # ---- helpers ----
    @property
    def step_dx(self):
        return self.base_dx * self.scale

    @property
    def step_dth(self):
        return self.base_dth * self.scale

    def _encode_last(self, model_frame):
        return self.lp.wm.encode_obs({"visual": model_frame})["visual"][:, -1:]

    def _decode_last(self, z_flat):
        return self.lp._decode_last(z_flat)      # [1,1,D] -> HWC uint8

    def _decode_traj(self, z_visual):
        """[1, T, D] latents -> list of HWC uint8 frames (decode_latents -> [0,1])."""
        T = z_visual.shape[1]
        lat = z_visual.reshape(1, T, self.lp.C_lat, self.lp.h_lat, self.lp.w_lat)
        frames = decode_latents(self.lp.vae, lat, self.lp.vae_precision).clamp(0, 1)
        return [(frames[0, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in range(T)]

    def _rollout(self):
        """Re-roll the WM under the full accumulated action list from the fixed start."""
        if not self.acts:
            return self._encode_last(self.start_model), self.start_rgb
        acts = torch.stack(self.acts).unsqueeze(0).to(self.device)        # [1,N,2]
        z, _ = self.lp.wm.rollout({"visual": self.start_model}, acts, num_sampling_steps=self.lp.ddim)
        z_last = z["visual"][:, -1:]
        return z_last, self._decode_last(z_last)

    def _dist(self, z_last):
        return None if self.zg is None else _flat_l2(z_last, self.zg)

    def _net(self):
        dx = sum(r[0] for r in self.raw) * 100.0          # cm
        dth = sum(r[1] for r in self.raw) * RAD2DEG        # deg
        return {"dx_cm": dx, "dth_deg": dth}

    # ---- actions (each returns a JSON-able dict; caller holds the lock) ----
    def set_goal(self, goal_model):
        self.goal_model = goal_model
        self.goal_rgb = to_uint8(goal_model[0, 0], model_range=self.normalize_pixel)
        self.zg = self._encode_last(goal_model)

    def step(self, key):
        if key not in KEY_ACTIONS:
            return {"error": f"unmapped key {key!r}"}
        fwd, turn = KEY_ACTIONS[key]
        raw = (fwd * self.step_dx, turn * self.step_dth)                  # (Δx m, Δθ rad)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.device)
        norm = (raw_t - self.lp.a_mean) / self.lp.a_std                   # WM consumes normalized actions
        self.acts.append(norm)
        self.raw.append(raw)
        z_last, cur_rgb = self._rollout()
        self.strip.append(cur_rgb)
        return {
            "step": len(self.acts), "scale": round(self.scale, 3),
            "last_action": {"dx_cm": raw[0] * 100, "dth_deg": math.degrees(raw[1]), "key": key},
            "net": self._net(), "dist": self._dist(z_last), "has_goal": self.zg is not None,
            "frame": png_b64(cur_rgb), "strip_append": png_b64(cur_rgb),
        }

    def undo(self):
        if self.acts:
            self.acts.pop()
            self.raw.pop()
            self.strip.pop()
        return self._full_state()

    def reset(self):
        self.acts, self.raw = [], []
        self.strip = [self.start_rgb]
        return self._full_state()

    def set_start(self, val_idx):
        """Swap the start frame to a different val slice (clears the current drive)."""
        val = self._build_val()
        if not (0 <= val_idx < len(val)):
            return {"error": f"start val {val_idx} out of range (0..{len(val) - 1})"}
        self.start_model = val[val_idx]["video"][:1].unsqueeze(0).to(self.device)   # dataset-native [-1,1]
        self.start_rgb = to_uint8(self.start_model[0, 0], model_range=self.normalize_pixel)
        self.start_val = int(val_idx)
        self.acts, self.raw = [], []
        self.strip = [self.start_rgb]
        return self._full_state()

    def rescale(self, direction):
        self.scale = max(0.1, min(8.0, self.scale * (1.25 if direction == "up" else 0.8)))
        return {"scale": round(self.scale, 3),
                "step_dx_cm": round(self.step_dx * 100, 2),
                "step_dth_deg": round(math.degrees(self.step_dth), 2)}

    def setgoal_current(self):
        """Snapshot the current driven frame as the new goal."""
        z_last, cur_rgb = self._rollout()
        self.set_goal(self._to_model(cur_rgb))
        st = self._full_state()
        st["goal"] = png_b64(self.goal_rgb)
        return st

    def cem(self):
        """Plan from the CURRENT driven frame toward the loaded goal; return the full imagined path."""
        if self.zg is None:
            return {"error": "no goal loaded — launch with --goal-* or press g to set one"}
        _, cur_rgb = self._rollout()
        obs_cur = {"visual": self._to_model(cur_rgb)}                     # observe the driven frame
        obs_goal = {"visual": self.goal_model}
        # CEM's internal rollouts read num_sampling_steps from the config (they don't pass it), so set
        # the cheap DDIM here — exactly like LekiwiPlanner.plan — else they fall back to the train default.
        self.lp._set_ddim(self.lp.ddim)
        planner = self.lp._make_planner()
        mu, info = planner.plan(obs_cur, obs_goal, return_elites=(self.lp.n_elite_viz > 0))
        mu = mu.detach().to(self.device)                                 # [1,H,2] normalized

        raw = (mu[0] * self.lp.a_std + self.lp.a_mean)                   # [H,2] (m, rad)
        suggested = [{"dx_cm": float(raw[t, 0]) * 100, "dth_deg": float(raw[t, 1]) * RAD2DEG}
                     for t in range(raw.shape[0])]

        z_cem, _ = self.lp.wm.rollout(obs_cur, mu, num_sampling_steps=self.lp.ddim)   # [1,1+H,D]
        traj = self._decode_traj(z_cem["visual"])                        # [start, t+1..t+H]
        reached = _flat_l2(z_cem["visual"][:, -1:], self.zg)

        elites = []
        ea = info.get("elite_actions")
        if ea is not None and self.lp.n_elite_viz > 0:
            for k in range(min(self.lp.n_elite_viz, ea.shape[0])):
                ze, _ = self.lp.wm.rollout(obs_cur, ea[k:k + 1].to(self.device),
                                           num_sampling_steps=self.lp.ddim)
                elites.append(png_b64(self._decode_last(ze["visual"][:, -1:])))
        return {
            "suggested": suggested,
            "final_loss": float(info.get("final_loss", 0.0)),
            "reached": reached, "do_nothing": self._dist(self._encode_last(obs_cur["visual"])),
            "traj": [png_b64(f) for f in traj], "elites": elites,
        }

    def _full_state(self):
        z_last, cur_rgb = self._rollout()
        return {
            "step": len(self.acts), "scale": round(self.scale, 3),
            "step_dx_cm": round(self.step_dx * 100, 2),
            "step_dth_deg": round(math.degrees(self.step_dth), 2),
            "net": self._net(), "dist": self._dist(z_last), "has_goal": self.zg is not None,
            "start_val": self.start_val, "n_val": self.n_val,
            "frame": png_b64(cur_rgb), "strip": [png_b64(f) for f in self.strip],
            "goal": png_b64(self.goal_rgb) if self.goal_rgb is not None else None,
        }


# ---------------------------------------------------------------------------
# Web server (stdlib; single user; one global Driver guarded by its lock).
# ---------------------------------------------------------------------------
DRIVER = None  # set in main()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>WM Driver</title>
<style>
 body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;margin:0;padding:14px}
 h2{margin:0 0 8px;font-weight:600;color:#9cf}
 .row{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .panel{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:10px}
 img.frame{width:320px;height:320px;image-rendering:pixelated;background:#000;border:1px solid #444}
 img.thumb{width:96px;height:96px;image-rendering:pixelated;background:#000;border:1px solid #444}
 .strip{display:flex;gap:4px;overflow-x:auto;max-width:96vw;padding-bottom:6px}
 .hud{font-size:13px;line-height:1.7} .hud b{color:#9cf}
 .k{display:inline-block;background:#222;border:1px solid #444;border-radius:4px;padding:1px 6px;margin:1px}
 #busy{color:#fc6} .dist{font-size:20px;color:#6f6} .err{color:#f66}
 small{color:#888}
 button{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;margin:2px;cursor:pointer;font-family:inherit}
 button:hover{background:#333}
 input[type=number]{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:2px 4px}
</style></head><body>
<h2>NanoWM interactive driver <small id="busy"></small></h2>
<div class="row">
  <div class="panel">
    <div><small>current (driven)</small></div>
    <img id="cur" class="frame">
    <div class="hud" id="hud"></div>
  </div>
  <div class="panel" id="goalpanel" style="display:none">
    <div><small>goal</small></div><img id="goal" class="frame">
    <div class="dist" id="dist"></div>
  </div>
  <div class="panel">
    <div class="hud">
      <b>drive</b> <span class="k">w</span>fwd <span class="k">s</span>back
      <span class="k">a</span>left <span class="k">d</span>right
      <span class="k">q</span>/<span class="k">e</span>arc<br>
      <span class="k">[</span>/<span class="k">]</span>step size
      <span class="k">c</span>CEM <span class="k">g</span>set goal=here
      <span class="k">u</span>undo <span class="k">r</span>reset
    </div>
  </div>
  <div class="panel">
    <div class="hud"><b>start frame</b> &nbsp;<span id="startinfo">val ?</span><br>
      <button onclick="startBy(-1)">◀ prev</button>
      <button onclick="startBy(1)">next ▶</button>
      <button onclick="startRand()">random</button><br>
      jump <input id="startjump" type="number" min="0" style="width:72px">
      <button onclick="startJump()">load</button>
      <div style="margin-top:6px"><small>switches the seed; clears your drive. keys <span class="k">,</span>/<span class="k">.</span> prev/next</small></div>
    </div>
  </div>
</div>
<div class="panel" style="margin-top:14px">
  <div><small>trajectory (your drive — watch drift accumulate)</small></div>
  <div class="strip" id="strip"></div>
</div>
<div class="panel" id="cempanel" style="margin-top:14px;display:none">
  <div><small>CEM imagined trajectory from current → goal &nbsp; <span id="cemhud"></span></small></div>
  <div class="strip" id="cemtraj"></div>
  <div><small>elite endpoints</small></div>
  <div class="strip" id="cemelites"></div>
</div>
<script>
let busy=false;
const $=id=>document.getElementById(id);
function setBusy(b){busy=b;$('busy').textContent=b?'… working':'';}
function thumb(src){const i=new Image();i.src=src;i.className='thumb';return i;}
function renderState(s){
  if(s.frame)$('cur').src=s.frame;
  if(s.goal){$('goalpanel').style.display='';$('goal').src=s.goal;}
  if('strip' in s){const st=$('strip');st.innerHTML='';s.strip.forEach(f=>st.appendChild(thumb(f)));}
  if(s.strip_append)$('strip').appendChild(thumb(s.strip_append));
  let h='<b>step</b> '+ (s.step??'-') +'  <b>scale</b> '+(s.scale??1)+'×'
       +' ('+(s.step_dx_cm??'?')+'cm / '+(s.step_dth_deg??'?')+'°)<br>';
  if(s.last_action)h+='<b>last</b> '+(s.last_action.key||'')+' Δx='+s.last_action.dx_cm.toFixed(2)
       +'cm Δθ='+s.last_action.dth_deg.toFixed(1)+'°<br>';
  if(s.net)h+='<b>net</b> Δx='+s.net.dx_cm.toFixed(1)+'cm Δθ='+s.net.dth_deg.toFixed(1)+'°';
  $('hud').innerHTML=h;
  if('dist' in s)$('dist').textContent = s.dist==null?'(no goal)':('latentL2 '+s.dist.toFixed(1));
  if('start_val' in s){window.START_VAL=s.start_val; window.N_VAL=s.n_val;
    $('startinfo').textContent = (s.start_val==null?'image seed':('val '+s.start_val))+' / '+s.n_val+' slices';
    if($('startjump'))$('startjump').max=(s.n_val-1);}
}
function startBy(d){let n=window.N_VAL||1, cur=window.START_VAL; if(cur==null)cur=0;
  call('/start?val='+((((cur+d)%n)+n)%n));}
function startRand(){let n=window.N_VAL||1; call('/start?val='+Math.floor(Math.random()*n));}
function startJump(){let v=parseInt($('startjump').value); if(!isNaN(v))call('/start?val='+v);}
function renderCem(c){
  if(c.error){$('cemhud').innerHTML='<span class="err">'+c.error+'</span>';$('cempanel').style.display='';return;}
  $('cempanel').style.display='';
  $('cemhud').textContent='reached='+c.reached.toFixed(1)+'  loss='+c.final_loss.toFixed(3)
     +'  | plan: '+c.suggested.map(a=>'Δx'+a.dx_cm.toFixed(1)+'/Δθ'+a.dth_deg.toFixed(0)).join('  ');
  const t=$('cemtraj');t.innerHTML='';c.traj.forEach(f=>t.appendChild(thumb(f)));
  const e=$('cemelites');e.innerHTML='';c.elites.forEach(f=>e.appendChild(thumb(f)));
}
async function call(path){
  if(busy)return; setBusy(true);
  try{const r=await fetch(path);const j=await r.json();
    if(path.startsWith('/cem'))renderCem(j); else renderState(j);
  }catch(e){console.error(e);} setBusy(false);
}
const MOVE=new Set(['w','a','s','d','q','e']);
document.addEventListener('keydown',ev=>{
  if(ev.target && ev.target.tagName==='INPUT')return;   // don't hijack typing in the jump box
  const k=ev.key.toLowerCase();
  if(MOVE.has(k))call('/step?key='+k);
  else if(k==='c')call('/cem');
  else if(k==='g')call('/setgoal');
  else if(k==='u')call('/undo');
  else if(k==='r')call('/reset');
  else if(k===']')call('/scale?dir=up');
  else if(k==='[')call('/scale?dir=down');
  else if(k===',')startBy(-1);
  else if(k==='.')startBy(1);
  else return;
  ev.preventDefault();
});
call('/state');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" or u.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        with DRIVER.lock:
            try:
                if u.path == "/state":
                    out = DRIVER._full_state()
                elif u.path == "/step":
                    out = DRIVER.step((q.get("key", [""])[0]).lower())
                elif u.path == "/undo":
                    out = DRIVER.undo()
                elif u.path == "/reset":
                    out = DRIVER.reset()
                elif u.path == "/start":
                    out = DRIVER.set_start(int(q.get("val", ["0"])[0]))
                elif u.path == "/scale":
                    out = DRIVER.rescale(q.get("dir", ["up"])[0])
                elif u.path == "/setgoal":
                    out = DRIVER.setgoal_current()
                elif u.path == "/cem":
                    out = DRIVER.cem()
                else:
                    self.send_error(404)
                    return
            except Exception as e:
                import traceback
                traceback.print_exc()
                out = {"error": f"{type(e).__name__}: {e}"}
        self._json(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/workspace/results/20260603_160326-NanoWM-B-2-F4S10-lekiwi/"
                                       "checkpoints/across_timesteps/epoch=13-step=8000.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ddim", type=int, default=3)
    # seeding (start required; goal optional)
    ap.add_argument("--start-val", type=int, default=None, help="held-out val slice index for the start frame")
    ap.add_argument("--start-image", default=None, help="PNG/JPG path for the start frame")
    ap.add_argument("--goal-val", type=int, default=None, help="val slice index for the goal (enables latentL2/CEM)")
    ap.add_argument("--goal-image", default=None, help="PNG/JPG path for the goal")
    # driving magnitudes (defaults ~ the dataset's full-speed chunk)
    ap.add_argument("--step-dx", type=float, default=0.0165, help="forward Δx per keystroke (m)")
    ap.add_argument("--step-dth-deg", type=float, default=9.0, help="turn Δθ per keystroke (deg)")
    # CEM (passed straight to LekiwiPlanner)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--opt-steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n-elite-viz", type=int, default=3)
    ap.add_argument("--action-mean", type=float, nargs=2, default=INTEGRATE_SE2_ACTION_MEAN)
    ap.add_argument("--action-std", type=float, nargs=2, default=INTEGRATE_SE2_ACTION_STD)
    args = ap.parse_args()

    if args.start_val is None and args.start_image is None:
        ap.error("provide a start frame: --start-val N or --start-image PATH")

    global DRIVER
    DRIVER = Driver(args)
    print(f"[driver] start={'val#%d' % args.start_val if args.start_val is not None else args.start_image} "
          f"goal={'val#%d' % args.goal_val if args.goal_val is not None else (args.goal_image or 'none')} "
          f"normalize_pixel={DRIVER.normalize_pixel} step_dx={args.step_dx*100:.2f}cm "
          f"step_dth={args.step_dth_deg:.1f}° DDIM={args.ddim}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[driver] serving on http://{args.host}:{args.port}  (forward the port, then open it in a browser)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[driver] bye")


if __name__ == "__main__":
    main()
