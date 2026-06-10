from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from .base import LatentShape


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DensePatchLatentCodec:
    has_decoder = False

    def __init__(
        self,
        model,
        latent_shape: LatentShape,
        input_size: int,
        patch_size: int,
        precision: str = "fp32",
        latent_scale: float = 1.0,
    ):
        self.model = model
        self.latent_shape = latent_shape
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.precision = precision
        # Divisor applied to encoder tokens so the diffusion/FM targets are ~unit-scale
        # (the SD-VAE analog is scaling_factor). DINOv2-small post-LN tokens measure
        # elementwise std ~2.4 on the lekiwi dataset -> latent_scale: 2.4 in config.
        self.latent_scale = float(latent_scale)
        self._device = torch.device("cpu")
        self._mean = None
        self._std = None

    def _buffers(self, device, dtype):
        if self._mean is None or self._mean.device != device or self._mean.dtype != dtype:
            self._mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
            self._std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        return self._mean, self._std

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected frames [B,C,H,W], got shape={tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB frames, got C={frames.shape[1]}")

        x = ((frames.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        mean, std = self._buffers(x.device, x.dtype)
        return (x - mean) / std

    def _tokens_to_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n_tokens, dim = tokens.shape
        expected_tokens = self.latent_shape.height * self.latent_shape.width
        if n_tokens != expected_tokens:
            raise RuntimeError(
                f"{self.kind} produced {n_tokens} tokens, expected {expected_tokens} "
                f"for latent shape {self.latent_shape.as_tuple()}"
            )
        if dim != self.latent_shape.channels:
            raise RuntimeError(
                f"{self.kind} produced token dim {dim}, expected {self.latent_shape.channels}"
            )
        lat = tokens.reshape(b, self.latent_shape.height, self.latent_shape.width, dim).permute(0, 3, 1, 2).contiguous()
        return lat / self.latent_scale

    def _autocast(self, device):
        device_type = device.type if device.type in ("cuda", "cpu") else "cuda"
        if self.precision == "bf16":
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)
        if self.precision == "fp32":
            return torch.autocast(device_type=device_type, enabled=False)
        if self.precision == "match_trainer":
            import contextlib

            return contextlib.nullcontext()
        raise ValueError(f"precision={self.precision!r}; expected fp32, bf16, or match_trainer")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.kind} is an encoder-only latent codec and cannot decode pixels")

    def requires_grad_(self, requires_grad: bool) -> "DensePatchLatentCodec":
        self.model.requires_grad_(requires_grad)
        return self

    def eval(self) -> "DensePatchLatentCodec":
        self.model.eval()
        return self

    def to(self, device) -> "DensePatchLatentCodec":
        device = torch.device(device)
        if self._device != device:
            self.model.to(device)
            self._device = device
        return self


class WebDINOLatentCodec(DensePatchLatentCodec):
    kind = "webdino"

    def __init__(self, model_path: str, latent_shape: LatentShape, input_size: int, patch_size: int, precision: str,
                 latent_scale: float = 1.0):
        from transformers import Dinov2Model

        model = Dinov2Model.from_pretrained(model_path).eval()
        super().__init__(
            model=model,
            latent_shape=latent_shape,
            input_size=input_size,
            patch_size=patch_size,
            precision=precision,
            latent_scale=latent_scale,
        )
        hidden_size = int(model.config.hidden_size)
        model_patch_size = int(model.config.patch_size)
        if hidden_size != latent_shape.channels:
            raise ValueError(f"Web-DINO hidden_size={hidden_size} does not match latent channels={latent_shape.channels}")
        if model_patch_size != self.patch_size:
            raise ValueError(f"Web-DINO patch_size={model_patch_size} does not match configured patch_size={self.patch_size}")

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self.to(frames.device)
        x = self._preprocess(frames)
        with self._autocast(x.device):
            outputs = self.model(pixel_values=x)
            tokens = outputs.last_hidden_state[:, 1:]
        return self._tokens_to_latents(tokens.float())


class VJEPA21LatentCodec(DensePatchLatentCodec):
    kind = "vjepa2_1"

    def __init__(
        self,
        checkpoint_path: str,
        latent_shape: LatentShape,
        input_size: int,
        patch_size: int,
        precision: str,
        repo_path: str | None = None,
        checkpoint_key: str = "ema_encoder",
        latent_scale: float = 1.0,
    ):
        encoder = self._load_encoder(repo_path=repo_path)
        checkpoint_file = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
        if checkpoint_key not in checkpoint:
            raise KeyError(f"Missing V-JEPA2.1 checkpoint key {checkpoint_key!r}; available keys: {list(checkpoint.keys())}")
        state_dict = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint[checkpoint_key].items()
        }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"V-JEPA2.1 encoder load mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        encoder.eval()
        super().__init__(
            model=encoder,
            latent_shape=latent_shape,
            input_size=input_size,
            patch_size=patch_size,
            precision=precision,
            latent_scale=latent_scale,
        )
        if int(encoder.embed_dim) != latent_shape.channels:
            raise ValueError(
                f"V-JEPA2.1 embed_dim={encoder.embed_dim} does not match latent channels={latent_shape.channels}"
            )

    @staticmethod
    def _load_encoder(repo_path: str | None):
        if repo_path:
            repo_root = str(Path(repo_path).resolve())
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            for module_name in list(sys.modules):
                if module_name == "src" or module_name.startswith("src."):
                    del sys.modules[module_name]
            project_root = Path(__file__).resolve().parents[2]
            removed_paths = []
            for path_entry in list(sys.path):
                try:
                    resolved = Path(path_entry or Path.cwd()).resolve()
                except OSError:
                    continue
                if resolved == project_root:
                    sys.path.remove(path_entry)
                    removed_paths.append(path_entry)
            try:
                from src.hub.backbones import vjepa2_1_vit_large_384

                encoder, _ = vjepa2_1_vit_large_384(pretrained=False)
            finally:
                for path_entry in reversed(removed_paths):
                    sys.path.insert(0, path_entry)
        else:
            encoder, _ = torch.hub.load(
                "facebookresearch/vjepa2",
                "vjepa2_1_vit_large_384",
                pretrained=False,
                trust_repo=True,
            )
        return encoder

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
        path = Path(str(checkpoint_path))
        if path.is_file():
            return path
        if path.is_dir():
            preferred = path / "vjepa2_1_vitl_dist_vitG_384.pt"
            if preferred.exists():
                return preferred
            candidates = sorted(path.glob("*.pt"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"V-JEPA2.1 checkpoint not found: {checkpoint_path}")

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self.to(frames.device)
        x = self._preprocess(frames).unsqueeze(2)  # [B,C,1,H,W], image tokenizer path
        with self._autocast(x.device):
            tokens = self.model(x)
        return self._tokens_to_latents(tokens.float())
