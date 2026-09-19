"""The two raster cleaners: the x4 upscaler (MambaIRv2) and the JPEG/WebP restorer (ConvNeXt).

Both are ports of the Inkvec reference implementations, kept numerically on the same recipe:

* Upscaler: ``tools/inkvec_sr/model.py`` and ``clean.py`` in the Inkvec repository
  (loading, the seeded forward pass, tiling with a feathered blend, alpha at Lanczos, and
  the x2 pre-pass recipe of box-halving plus recolouring the flats).
* Restorer: ``crates/inkvec-restore/src/lib.rs`` and ``planar.rs`` (composite onto white,
  replicate-pad to a multiple of 16, quantise to 8-bit levels, snap near-extremes, carry
  alpha through), with ``crates/inkvec-sr/src/detect.rs`` for the ``auto`` decision.

Heavy dependencies (einops, onnxruntime, huggingface_hub) are imported inside functions so
the trace nodes load without them. Arrays here are numpy float32 in [0, 1].
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("inkvec-comfyui")


class CleanerUnavailable(RuntimeError):
    """A cleaner cannot run here; the message says what to install or fix."""


# ============================================================================ weights
@dataclass(frozen=True)
class HubFile:
    repo_id: str
    revision: str  # a pinned commit, so the vendored code and the weights cannot drift apart
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.filename}"

    @property
    def cache_dir(self) -> Path:
        return models_root() / self.repo_id.split("/")[-1]


SR_WEIGHTS = HubFile(
    "Logolabs/inkvec-sr-001",
    "a16b17451224c9a51a2d6bb94459d96d18eb5267",
    "logo_sr_x4.pt",
    "fff0131eb5246c0d420d37a28181d8c7128b06a6ab937febc800505d4956a4f9",
    20426992,
)
DENOISER_WEIGHTS = HubFile(
    "Logolabs/inkvec-denoiser-001",
    "f0adf499be977be70df3c4015e443aeed4a3d52f",
    "restorer.onnx",
    # Same value as EXPECTED_SHA256 in crates/inkvec-restore/src/lib.rs and the repo's SHA256SUMS.
    "bdc2762157632f6f74dd91474f0598e591d49ded47e0a87642c89416b0809d6d",
    79943513,
)

_verified: set[tuple[str, int, float]] = set()
_fetch_lock = threading.Lock()


def models_root() -> Path:
    """``<ComfyUI>/models/inkvec`` inside ComfyUI, else a per-user cache directory."""
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.models_dir) / "inkvec"
    except Exception:
        pass
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "inkvec-comfyui"


def _is_verified(path: Path, f: HubFile) -> bool:
    from .binary_manager import sha256_file

    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key in _verified:
        return True
    if st.st_size == f.size and sha256_file(path) == f.sha256:
        _verified.add(key)
        return True
    return False


def _download_https(f: HubFile, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(f.url, headers={"User-Agent": "inkvec-comfyui"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as fh:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                fh.write(chunk)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def fetch(f: HubFile, override: str = "") -> Path:
    """Local path of a weights file: `override` if given (used as-is, not hash-checked),
    else the cached copy, else a download pinned to `f.revision` and SHA-256 verified."""
    if override and override.strip():
        p = Path(override.strip()).expanduser()
        if p.is_dir():
            p = p / f.filename
        if not p.is_file():
            raise CleanerUnavailable(f"weights_path {override!r}: no {f.filename} there")
        return p
    dest = f.cache_dir / f.filename
    with _fetch_lock:
        if dest.is_file():
            if _is_verified(dest, f):
                return dest
            log.warning("[inkvec] %s does not match its published SHA-256; downloading it again", dest)
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("[inkvec] downloading %s from %s (%.0f MB)", f.filename, f.repo_id, f.size / 1e6)
        try:
            from huggingface_hub import hf_hub_download

            got = Path(hf_hub_download(f.repo_id, f.filename, revision=f.revision, local_dir=str(f.cache_dir)))
            if got.resolve() != dest.resolve():
                os.replace(got, dest)
        except ImportError:
            _download_https(f, dest)
        except Exception as hub_exc:  # hub errors: fall back to the plain URL once
            log.warning("[inkvec] huggingface_hub download failed (%s); trying %s", hub_exc, f.url)
            try:
                _download_https(f, dest)
            except Exception as exc:
                raise CleanerUnavailable(
                    f"could not download {f.filename} from {f.url}: {exc}. Download it by hand into "
                    f"{dest.parent} or set weights_path."
                ) from exc
        if not _is_verified(dest, f):
            dest.unlink(missing_ok=True)
            raise CleanerUnavailable(
                f"{f.filename} downloaded from {f.repo_id} failed SHA-256 verification; not using it"
            )
        return dest


def pick_torch_device(choice: str) -> str:
    import torch

    if choice == "cuda" and not torch.cuda.is_available():
        raise CleanerUnavailable("device is set to cuda, but this PyTorch build sees no CUDA device")
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


# ============================================================================ upscaler
#: Fixed RNG seed for MambaIRv2's Gumbel-softmax routing ("VAC1"), as in the reference.
ROUTING_SEED = 0x5641_4331
#: The model's window size; a partial window is not valid input.
SR_PAD = 16


@dataclass
class Upscaler:
    net: object
    scale: int
    device: str
    backend: str
    step: int | None = None

    @property
    def label(self) -> str:
        return f"MambaIRv2 x{self.scale} step {self.step} on {self.device} [{self.backend}]"


_upscalers: dict[tuple[str, str], Upscaler] = {}
_sr_lock = threading.Lock()


def load_upscaler(device: str = "auto", weights_path: str = "") -> Upscaler:
    """Load (once per weights file and device) the x4 upscaler."""
    try:
        import torch
        import einops  # noqa: F401  (the architecture needs it)
    except ImportError as exc:
        raise CleanerUnavailable(
            f"the upscaler needs PyTorch and einops ({exc}). Install with: pip install einops"
        ) from exc
    dev = pick_torch_device(device)
    weights = fetch(SR_WEIGHTS, weights_path)
    key = (str(weights), dev)
    with _sr_lock:
        if key in _upscalers:
            return _upscalers[key]
        from .vendor import scan
        from .vendor.mambairv2_arch import MambaIRv2

        # weights_only: the checkpoint is plain dicts, lists, numbers and tensors, so the
        # restricted unpickler loads it and no code in the file can run.
        blob = torch.load(weights, map_location="cpu", weights_only=True)
        # Constructing the network draws its (then overwritten) random init from the global
        # RNG; fork it so loading does not move ComfyUI's RNG state.
        with torch.random.fork_rng(devices=[]):
            net = MambaIRv2(**blob["arch_kwargs"])
        missing, _ = net.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()}, strict=False)
        if missing:
            raise CleanerUnavailable(f"checkpoint does not match the architecture: {len(missing)} missing tensors")
        up = Upscaler(net.to(dev).eval(), int(blob["scale"]), dev, scan.backend(dev), blob.get("trained_step"))
        _upscalers[key] = up
        log.info("[inkvec] loaded %s", up.label)
        return up


def _seeded(torch, fn, *a, **kw):
    """Run `fn` with the global RNG pinned to ROUTING_SEED, then put the RNG back.

    MambaIRv2's ASSM routes pixels with `F.gumbel_softmax(logits, hard=True)`, which samples
    noise on every call, including under eval and no_grad; unseeded, repeated runs of one
    image differ by up to 12.45 levels. The argmax limit is deterministic but measured worse
    (dE00 0.5492 against 0.5364), so the reference pins one draw instead.
    """
    state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(ROUTING_SEED)
        return fn(*a, **kw)
    finally:
        torch.random.set_rng_state(state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def upscale(up: Upscaler, rgb: np.ndarray, tile: int = 256, overlap: int = 32,
            bf16: bool = True, progress=None) -> np.ndarray:
    """rgb float HxWx3 in [0, 1] -> upscaled by `up.scale`, same range (reference `upscale`)."""
    import torch

    with torch.no_grad():
        h, w, _ = rgb.shape
        s = up.scale
        pad = SR_PAD
        out = np.zeros((h * s, w * s, 3), np.float32)
        acc = np.zeros((h * s, w * s, 1), np.float32)
        step = max(1, tile - overlap)
        ys = list(range(0, max(1, h - overlap), step)) or [0]
        xs = list(range(0, max(1, w - overlap), step)) or [0]
        for y in ys:
            for x in xs:
                y1, x1 = min(y + tile, h), min(x + tile, w)
                y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
                patch = torch.from_numpy(
                    np.ascontiguousarray(rgb[y0:y1, x0:x1], np.float32)
                ).permute(2, 0, 1)[None].to(up.device)
                ph_, pw_ = patch.shape[-2:]
                py, px = (-ph_) % pad, (-pw_) % pad
                if py or px:
                    # Reflect as the reference does; reflect needs the pad to be smaller than
                    # the side, so only an image under ~9 px falls back to replicate.
                    mode = "reflect" if py < ph_ and px < pw_ else "replicate"
                    patch = torch.nn.functional.pad(patch, (0, px, 0, py), mode=mode)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=bf16 and str(up.device).startswith("cuda")):
                    pred = _seeded(torch, up.net, patch)
                pred = pred.float().clamp(0, 1)[..., : ph_ * s, : pw_ * s]
                pred = pred[0].permute(1, 2, 0).cpu().numpy()
                ph, pw = pred.shape[:2]
                # Feather, so a tile seam does not become a traced edge.
                wy = np.minimum(np.arange(ph), ph - 1 - np.arange(ph)) + 1.0
                wx = np.minimum(np.arange(pw), pw - 1 - np.arange(pw)) + 1.0
                wgt = np.minimum(wy[:, None], wx[None, :])[..., None].astype(np.float32)
                out[y0 * s:y0 * s + ph, x0 * s:x0 * s + pw] += pred * wgt
                acc[y0 * s:y0 * s + ph, x0 * s:x0 * s + pw] += wgt
                if progress is not None:
                    progress()
        return out / np.maximum(acc, 1e-6)


def sr_tile_count(h: int, w: int, tile: int, overlap: int = 32) -> int:
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step)) or [0]
    xs = list(range(0, max(1, w - overlap), step)) or [0]
    return len(ys) * len(xs)


def upscale_rgba(up: Upscaler, rgba: np.ndarray, **kw) -> np.ndarray:
    """HxWx4 in [0, 1] -> upscaled by `up.scale`, alpha carried at Lanczos (reference)."""
    from PIL import Image

    alpha, rgb = rgba[..., 3:4], rgba[..., :3]
    # The model sees colour, not colour x alpha, so a fully transparent pixel does not drag
    # its neighbours dark (which a tracer would pick up as a spurious contour).
    rgb = np.where(alpha > 1e-4, rgb, 0.0)
    hi = upscale(up, rgb, **kw)
    s = up.scale
    a_up = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8)).resize(
        (alpha.shape[1] * s, alpha.shape[0] * s), Image.LANCZOS)
    return np.dstack([np.clip(hi, 0, 1), np.asarray(a_up, np.float32)[..., None] / 255.0])


# --- the x2 pre-pass recipe (tools/inkvec_sr/clean.py, unchanged) -------------------------
def box_downsample(a: np.ndarray, factor: int = 2) -> np.ndarray:
    """Exact box average by an integer factor."""
    f = factor
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:h, :w]
    return b.reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))


def _flat_mask(img: np.ndarray, tol: float) -> np.ndarray:
    """True where the 3x3 neighbourhood spans less than `tol`."""
    g = img.mean(axis=2)
    lo = hi = g
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r = np.roll(np.roll(g, dy, 0), dx, 1)
            lo, hi = np.minimum(lo, r), np.maximum(hi, r)
    flat = (hi - lo) < tol
    flat[:1, :] = flat[-1:, :] = flat[:, :1] = flat[:, -1:] = False
    return flat


def match_flats(hi: np.ndarray, src: np.ndarray, tol: float = 3.0 / 255.0) -> np.ndarray:
    """Put the upscaler's flat regions back on the source's colours: one affine map per
    channel, fitted over the pixels a bicubic upsample of the input says are flat."""
    from PIL import Image

    bi = np.asarray(
        Image.fromarray((src.clip(0, 1) * 255).astype(np.uint8)).resize(
            (hi.shape[1], hi.shape[0]), Image.BICUBIC), np.float64) / 255.0
    flat = _flat_mask(bi, tol)
    if flat.mean() < 0.01:  # nothing flat enough to fit on; use everything
        flat = np.ones(bi.shape[:2], bool)

    out = hi.astype(np.float64).copy()
    for c in range(3):
        x, y = hi[..., c][flat], bi[..., c][flat]
        if x.size < 16 or x.std() < 1e-6:
            continue
        A = np.stack([x, np.ones_like(x)], axis=1)
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        out[..., c] = hi[..., c] * a + b
    return out.clip(0, 1)


def prepass(up: Upscaler, rgba: np.ndarray, out_scale: int = 2, recolour: bool = True, **kw) -> np.ndarray:
    """RGBA in [0, 1] -> cleaned RGBA at `out_scale` times the input size (reference)."""
    hi = upscale_rgba(up, rgba, **kw)
    factor = up.scale // out_scale
    if factor > 1:
        hi = box_downsample(hi, factor)
    elif factor < 1:
        raise ValueError(f"cannot output x{out_scale} from an x{up.scale} model")
    if recolour:
        # Fit against the RGB the model actually saw (zeroed where transparent).
        src = np.where(rgba[..., 3:4] > 1e-4, rgba[..., :3], 0.0)
        hi = np.dstack([match_flats(hi[..., :3], src), hi[..., 3:4]])
    return hi.clip(0, 1).astype(np.float32)


# ============================================================================ restorer
#: Both sides of the network input must be multiples of this (four 2x downsamplings).
MULTIPLE = 16
#: Pixels within this many /255 levels of an extreme, on every channel, snap to it.
SNAP_LEVELS = 6
#: Interior residual above which `auto` restores (inkvec_sr::detect::DEGRADED_RESIDUAL).
DEGRADED_RESIDUAL = 0.5
#: Tile overlap for the optional tiled mode (tools/restore.py and the model card: 128 / 32).
DENOISE_OVERLAP = 32

_sessions: dict[tuple[str, str], object] = {}
_ort_lock = threading.Lock()


@dataclass
class Restorer:
    session: object
    providers: list

    @property
    def label(self) -> str:
        return f"ConvNeXt restorer (ONNX Runtime, {', '.join(self.providers)})"


def load_restorer(device: str = "auto", weights_path: str = "") -> Restorer:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise CleanerUnavailable(
            "the denoiser needs ONNX Runtime. Install with: pip install onnxruntime "
            "(or onnxruntime-gpu for CUDA)"
        ) from exc
    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise CleanerUnavailable(
                f"device is set to cuda, but ONNX Runtime offers only {available}; "
                f"install onnxruntime-gpu or choose cpu"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "auto" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    weights = fetch(DENOISER_WEIGHTS, weights_path)
    key = (str(weights), ",".join(providers))
    with _ort_lock:
        if key not in _sessions:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # On CUDA, cuDNN's autotuned convolutions differ run to run (measured 2.6e-4 on
            # this network, enough to flip a pixel across the snap threshold); deterministic
            # compute makes repeated runs bit-identical. It changes nothing on CPU.
            opts.use_deterministic_compute = True
            sess = ort.InferenceSession(str(weights), sess_options=opts, providers=providers)
            used = sess.get_providers()
            if device == "cuda" and used[0] != "CUDAExecutionProvider":
                raise CleanerUnavailable(
                    "ONNX Runtime could not start its CUDA provider (it needs CUDA 12 and cuDNN 9 "
                    "on the library path); choose cpu, or fix the CUDA install"
                )
            _sessions[key] = Restorer(sess, used)
            log.info("[inkvec] loaded %s", _sessions[key].label)
        return _sessions[key]


def _run_padded(r: Restorer, rgb: np.ndarray) -> np.ndarray:
    """planar.rs: replicate-pad bottom/right to a multiple of 16, run, crop back."""
    h, w, _ = rgb.shape
    ph, pw = -(-h // MULTIPLE) * MULTIPLE, -(-w // MULTIPLE) * MULTIPLE
    padded = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    (y,) = r.session.run(["restored"], {"image": chw})
    return y[0].transpose(1, 2, 0)[:h, :w]


def _run_tiled(r: Restorer, rgb: np.ndarray, tile: int, overlap: int = DENOISE_OVERLAP) -> np.ndarray:
    """tools/restore.py's tiling: overlapping tiles, averaged with equal weight. Each tile is
    padded like a whole image, so tiles at the edge of a small image are valid input too."""
    h, w, _ = rgb.shape
    step = max(1, tile - overlap)
    out = np.zeros((h, w, 3), np.float32)
    acc = np.zeros((h, w, 1), np.float32)
    for y0 in range(0, max(h - overlap, 1), step):
        for x0 in range(0, max(w - overlap, 1), step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0b, x0b = max(y1 - tile, 0), max(x1 - tile, 0)
            out[y0b:y1, x0b:x1] += _run_padded(r, rgb[y0b:y1, x0b:x1])
            acc[y0b:y1, x0b:x1] += 1.0
    return out / np.maximum(acc, 1.0)


def quantize_levels(rgb: np.ndarray) -> np.ndarray:
    """Round to the nearest of 256 levels, clamped to [0, 1] (lib.rs `quantize_levels`).

    In f32 like the Rust code, and with Rust's rounding (ties away from zero) rather than
    numpy's ties-to-even, so a value exactly halfway between two levels lands where it does
    in the engine."""
    x = np.clip(rgb.astype(np.float32), np.float32(0.0), np.float32(1.0)) * np.float32(255.0)
    fl = np.floor(x)
    return (np.where(x - fl >= np.float32(0.5), fl + np.float32(1.0), fl) / np.float32(255.0)).astype(np.float32)


def snap_extremes(rgb: np.ndarray) -> np.ndarray:
    """Every channel within SNAP_LEVELS/255 of 1 (or 0) -> exactly 1 (or 0) (lib.rs)."""
    t = np.float32(SNAP_LEVELS / 255.0)
    rgb = rgb.astype(np.float32)
    hi = (rgb >= np.float32(1.0) - t).all(axis=-1, keepdims=True)
    lo = (rgb <= t).all(axis=-1, keepdims=True)
    return np.where(hi, np.float32(1.0), np.where(lo, np.float32(0.0), rgb))


def on_white(rgba: np.ndarray) -> np.ndarray:
    """Straight-alpha RGBA -> RGB composited onto white."""
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + (1.0 - a)


def restore_rgba(r: Restorer, rgba: np.ndarray, tile: int = 0) -> np.ndarray:
    """lib.rs `restore_rgba`: composite onto white (the network is RGB-only and was trained on
    opaque renders), restore, quantise, snap, and carry alpha through unchanged."""
    rgb = on_white(rgba).astype(np.float32)
    out = _run_tiled(r, rgb, tile) if tile and tile > 0 else _run_padded(r, rgb)
    out = snap_extremes(quantize_levels(out))
    return np.dstack([out, rgba[..., 3:4]]).astype(np.float32)


def flat_mask(rgb: np.ndarray, tol: float) -> np.ndarray:
    """clean.rs `flat_mask`: interior pixels whose 3x3 luma range is below `tol`."""
    g = rgb.mean(axis=2)
    h, w = g.shape
    mask = np.zeros((h, w), bool)
    if h < 3 or w < 3:
        return mask
    win = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    mask[1:-1, 1:-1] = (win.max(axis=(-1, -2)) - win.min(axis=(-1, -2))) < tol
    return mask


def interior_residual(inp_rgba: np.ndarray, model_rgba: np.ndarray) -> float | None:
    """detect.rs `interior_residual`: RMS disagreement in 8-bit levels where the (composited)
    trace is flat, with the calibrated sum/9n normalisation. None below 100 flat pixels."""
    if inp_rgba.shape[:2] != model_rgba.shape[:2]:
        return None
    a, b = on_white(inp_rgba), on_white(model_rgba)
    mask = flat_mask(b, np.float32(1.5 / 255.0))
    n = int(mask.sum())
    if n < 100:
        return None
    d = (b[mask] - a[mask]).astype(np.float64) * 255.0
    return float(np.sqrt((d * d).sum() / (n * 9.0)))
