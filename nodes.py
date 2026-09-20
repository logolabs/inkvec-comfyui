"""ComfyUI nodes for Inkvec: trace to SVG, save SVG, and the two optional raster cleaners.

Tensor conventions (ComfyUI): IMAGE is float [B, H, W, C] in 0..1, usually C = 3; MASK is
float [B, H, W] with 1 = masked. LoadImage drops alpha from IMAGE and returns it as
MASK = 1 - alpha, so here an optional MASK input is read as alpha = 1 - mask, and a
4-channel IMAGE is accepted as RGBA directly. The MASK outputs follow the same convention.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import binary_manager, options, tracer

log = logging.getLogger("inkvec-comfyui")

CATEGORY = "Inkvec"


# ============================================================================ helpers
def split_image(image: torch.Tensor, mask: torch.Tensor | None = None):
    """IMAGE (+ optional MASK) -> (rgb [B,H,W,3], alpha [B,H,W] or None), float32 on CPU.

    A MASK input wins over a fourth channel. A mask of another size is resized to the image
    (LoadImage returns a 64x64 zero mask for images without alpha); a mask batch shorter
    than the image batch repeats its last entry.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"expected an IMAGE tensor, got {type(image).__name__}")
    img = image.detach().to("cpu", torch.float32)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4 or img.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected an IMAGE tensor [B,H,W,C] with C = 1, 3 or 4, got {tuple(image.shape)}")
    b, h, w, c = img.shape
    rgb = img[..., :3] if c >= 3 else img.expand(b, h, w, 3)
    alpha = img[..., 3] if c == 4 else None
    if mask is not None:
        m = mask.detach().to("cpu", torch.float32)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() != 3:
            raise ValueError(f"expected a MASK tensor [B,H,W], got {tuple(mask.shape)}")
        if tuple(m.shape[-2:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        if m.shape[0] != b:
            m = m[torch.clamp(torch.arange(b), max=m.shape[0] - 1)]
        alpha = 1.0 - m
    if alpha is not None:
        alpha = alpha.clamp(0, 1).contiguous()
    return rgb.clamp(0, 1).contiguous(), alpha


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()


def _rgba_array(rgb: torch.Tensor, alpha: torch.Tensor | None, i: int) -> np.ndarray:
    """Frame i as float32 HxWx4 straight-alpha RGBA."""
    a = alpha[i].numpy() if alpha is not None else np.ones(rgb.shape[1:3], np.float32)
    return np.dstack([rgb[i].numpy(), a]).astype(np.float32)


def _stack(frames: list[torch.Tensor]) -> torch.Tensor:
    """Batch frames of [H,W,...]; frames of another size are resized to the first one's."""
    h, w = frames[0].shape[:2]
    out = []
    for f in frames:
        if tuple(f.shape[:2]) != (h, w):
            x = f.unsqueeze(0) if f.dim() == 2 else f.permute(2, 0, 1)
            x = F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
            f = x[0] if f.dim() == 2 else x.permute(1, 2, 0)
        out.append(f)
    return torch.stack(out).contiguous()


def _temp_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        d = Path(folder_paths.get_temp_directory())
    except Exception:
        d = Path(tempfile.gettempdir()) / "inkvec-comfyui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd() / "output"


def _save_path(prefix: str, out_dir: Path):
    """folder_paths.get_save_image_path when running in ComfyUI; the same scheme otherwise."""
    try:
        import folder_paths  # type: ignore

        fn = folder_paths.get_save_image_path
    except (ImportError, AttributeError):
        fn = None
    if fn is not None:
        return fn(prefix, str(out_dir))
    out_dir = Path(out_dir).resolve()
    subfolder = os.path.dirname(os.path.normpath(prefix))
    name = os.path.basename(os.path.normpath(prefix))
    folder = (out_dir / subfolder).resolve()
    if os.path.commonpath([str(out_dir), str(folder)]) != str(out_dir):
        raise ValueError(f"filename_prefix {prefix!r} points outside the output directory")
    folder.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(name)}_(\d+)_", re.IGNORECASE)
    counts = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))]
    return str(folder), name, max(counts, default=0) + 1, subfolder, prefix


def _comfy_interrupt():
    """(is_interrupted, raise_if_interrupted) from ComfyUI, or no-ops outside it."""
    try:
        import comfy.model_management as mm  # type: ignore

        return mm.processing_interrupted, mm.throw_exception_if_processing_interrupted
    except Exception:
        return (lambda: False), (lambda: None)


def _progress(total: int):
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.ProgressBar(total)
    except Exception:
        return None


def _write_png(path: Path, rgb8: np.ndarray, a8: np.ndarray | None) -> None:
    arr = rgb8 if a8 is None else np.dstack([rgb8, a8])
    Image.fromarray(np.ascontiguousarray(arr)).save(path, compress_level=1)


def _trace_frame(rgb8: np.ndarray, a8: np.ndarray | None, args: list[str], timeout_sec: float) -> str:
    """Write one frame to a temp PNG, trace it, clean up. Honours ComfyUI's interrupt."""
    stop, raise_interrupt = _comfy_interrupt()
    tmp = _temp_dir()
    stem = f"inkvec-{uuid.uuid4().hex[:12]}"
    png, svg = tmp / f"{stem}.png", tmp / f"{stem}.svg"
    try:
        _write_png(png, rgb8, a8)
        svg_text, _ = tracer.run_inkvec(png, svg, args, timeout_sec=timeout_sec, should_stop=stop)
    except tracer.TraceInterrupted:
        raise_interrupt()
        raise
    finally:
        for p in (png, svg):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    return svg_text


_noted_no_renderer = False


def _note_no_renderer() -> None:
    global _noted_no_renderer
    if not _noted_no_renderer:
        _noted_no_renderer = True
        log.warning(
            "[inkvec] no SVG renderer installed, so the preview and mask outputs of Inkvec Trace "
            "pass the input through instead of showing the trace. `pip install resvg-py` "
            "(or cairosvg) enables them."
        )


# ============================================================================ trace
class InkvecTrace:
    """Trace an IMAGE to SVG with the native inkvec binary."""

    DESCRIPTION = (
        "Trace an image to SVG with the Inkvec tracer. Runs the native inkvec binary, which is "
        "downloaded from github.com/logolabs/inkvec on first use. Connect LoadImage's MASK to "
        "carry transparency."
    )
    CATEGORY = CATEGORY
    FUNCTION = "trace"
    RETURN_TYPES = ("STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("svg", "preview", "mask")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_TOOLTIPS = (
        "The SVG source, one string per image in the batch.",
        "The SVG rendered back to pixels (needs resvg-py or cairosvg; otherwise the input image).",
        "Transparency of the rendered SVG, 1 = transparent (LoadImage's convention).",
    )
    SEARCH_ALIASES = ["vectorize", "vectorise", "image to svg", "raster to svg", "trace"]

    @classmethod
    def INPUT_TYPES(cls):
        # Every tracer option comes from the one table in options.py (a schema shipped with
        # the binary, or the built-in table); only the node's own inputs are declared here.
        required = {"image": ("IMAGE",)}
        for opt in options.table():
            required[opt.name] = options.widget(opt)
        required["timeout_sec"] = ("INT", {
            "default": 300, "min": 5, "max": 7200,
            "tooltip": "The node stops the tracer after this many seconds and reports an error.",
        })
        return {
            "required": required,
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Transparency, as LoadImage outputs it (1 = transparent). A mask where "
                               "1 marks the subject must be inverted first (InvertMask).",
                }),
                "extra_args": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Further inkvec flags, appended last so they override the widgets, e.g. "
                               "--strokes or --layers. A flag the installed binary does not know is "
                               "reported as an error. -o/--output/--help/--version are not allowed.",
                }),
            },
        }

    def trace(self, image, timeout_sec=300, mask=None, extra_args="", **values):
        tracer.parse_extra_args(extra_args)  # fail before any work on a malformed string
        rgb, alpha = split_image(image, mask)
        n = rgb.shape[0]
        pbar = _progress(n)
        svgs, previews, masks = [], [], []
        for i in range(n):
            rgb8 = _to_uint8(rgb[i])
            a8 = _to_uint8(alpha[i]) if alpha is not None else None
            if a8 is not None and int(a8.min()) == 255:
                a8 = None  # fully opaque: trace it as RGB
            args = tracer.build_args(values, has_alpha=a8 is not None, extra_args=extra_args)
            try:
                svg_text = _trace_frame(rgb8, a8, args, timeout_sec)
            except (tracer.InkvecError, binary_manager.BinaryError) as exc:
                where = f" (image {i + 1} of {n})" if n > 1 else ""
                raise RuntimeError(f"Inkvec Trace{where}: {exc}") from exc
            svgs.append(svg_text)
            rendered = tracer.render_svg(svg_text)
            if rendered is None:
                _note_no_renderer()
                previews.append(rgb[i])
                masks.append(1.0 - alpha[i] if alpha is not None else torch.zeros(rgb.shape[1:3]))
            else:
                t = torch.from_numpy(rendered)
                previews.append(t[..., :3])
                masks.append(1.0 - t[..., 3])
            if pbar is not None:
                pbar.update(1)
        return (svgs, _stack(previews), _stack(masks))


class InkvecSaveSVG:
    """Write SVG text to ComfyUI's output directory."""

    DESCRIPTION = "Save SVG text to the ComfyUI output folder as <prefix>_00001_.svg, numbered like Save Image."
    CATEGORY = CATEGORY
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg": ("STRING", {"forceInput": True, "tooltip": "SVG source, e.g. from Inkvec Trace."}),
                "filename_prefix": ("STRING", {
                    "default": "inkvec",
                    "tooltip": "File name prefix; may include a subfolder, e.g. logos/inkvec.",
                }),
            }
        }

    def save(self, svg, filename_prefix="inkvec"):
        if not isinstance(svg, str) or "<svg" not in svg:
            raise ValueError("Inkvec Save SVG: the input is not SVG text")
        out_dir = _output_dir()
        folder, name, counter, subfolder, _ = _save_path(filename_prefix, out_dir)
        filename = f"{name}_{counter:05}_.svg"
        path = Path(folder) / filename
        path.write_text(svg, encoding="utf-8")
        log.info("[inkvec] saved %s", path)
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


# ============================================================================ cleaners
_DEVICE = (["auto", "cpu", "cuda"], {
    "default": "auto",
    "tooltip": "auto = CUDA when available, else CPU.",
})


def _cleaner_error(node: str, exc: Exception) -> RuntimeError:
    return RuntimeError(f"{node}: {exc}")


class InkvecUpscale:
    """x4 super-resolution for logos and flat art (MambaIRv2, Logolabs/inkvec-sr-001)."""

    DESCRIPTION = (
        "x4 upscaler for logos, icons and flat artwork (MambaIRv2-Small fine-tune, "
        "Logolabs/inkvec-sr-001), for small or blurred inputs before tracing. Weights "
        "download on first use. Deterministic: the routing RNG is pinned."
    )
    CATEGORY = CATEGORY
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": (["4x", "2x"], {
                    "default": "4x",
                    "tooltip": "4x = the network's output. 2x = the recipe `inkvec --sr on` traces: "
                               "x4, box-averaged back to x2, flat colours refitted to the source's.",
                }),
                "device": _DEVICE,
                "tile": ("INT", {
                    "default": 256, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Tile size in input pixels (32 px overlap, feathered blend). Lower it "
                               "if the GPU runs out of memory.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Transparency, 1 = transparent (LoadImage's MASK)."}),
                "weights_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local logo_sr_x4.pt (or its folder) instead of the download. Empty = "
                               "download from Hugging Face into models/inkvec/.",
                }),
            },
        }

    def upscale(self, image, scale, device, tile, mask=None, weights_path=""):
        from . import cleaners

        try:
            up = cleaners.load_upscaler(device, weights_path)
        except cleaners.CleanerUnavailable as exc:
            raise _cleaner_error("Inkvec Upscale", exc) from exc
        rgb, alpha = split_image(image, mask)
        n, h, w = rgb.shape[:3]
        out_scale = 4 if scale == "4x" else 2
        _, raise_interrupt = _comfy_interrupt()
        pbar = _progress(n * cleaners.sr_tile_count(h, w, int(tile)))

        def tick():
            if pbar is not None:
                pbar.update(1)
            raise_interrupt()

        images, masks = [], []
        for i in range(n):
            rgba = _rgba_array(rgb, alpha, i)
            if out_scale == up.scale:
                hi = cleaners.upscale_rgba(up, rgba, tile=int(tile), progress=tick)
            else:
                hi = cleaners.prepass(up, rgba, out_scale=out_scale, recolour=True, tile=int(tile), progress=tick)
            t = torch.from_numpy(np.ascontiguousarray(hi, dtype=np.float32))
            images.append(t[..., :3])
            masks.append(1.0 - t[..., 3])
        return (_stack(images), _stack(masks))


class InkvecDenoise:
    """Remove JPEG/WebP/decoder damage from flat art (ConvNeXt U-Net, Logolabs/inkvec-denoiser-001)."""

    DESCRIPTION = (
        "Restorer for JPEG, WebP and AI-decoder damage on logos and flat artwork (ConvNeXt U-Net, "
        "Logolabs/inkvec-denoiser-001), run like `inkvec --restore`. Weights download on first "
        "use. auto needs the inkvec binary and a rendered trace (resvg-py)."
    )
    CATEGORY = CATEGORY
    FUNCTION = "denoise"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["auto", "on"], {
                    "default": "auto",
                    "tooltip": "on = always restore. auto = trace once, and restore only if the input "
                               "disagrees with its own trace where the trace is flat (interior residual "
                               "above 0.5, the threshold of `inkvec --restore auto`); otherwise the "
                               "image passes through unchanged. On clean input the restorer costs a "
                               "few percent of colour accuracy.",
                }),
                "device": _DEVICE,
                "tile": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 16,
                    "tooltip": "0 = whole image at once, as the Inkvec engine runs it. 128 = the tiled "
                               "inference (32 px overlap) the model card's accuracy numbers were "
                               "measured with; it also bounds memory on large images.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Transparency, 1 = transparent (LoadImage's MASK). Passed through."}),
                "weights_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local restorer.onnx (or its folder) instead of the download. Empty = "
                               "download from Hugging Face into models/inkvec/.",
                }),
            },
        }

    def denoise(self, image, mode, device, tile, mask=None, weights_path=""):
        from . import cleaners

        if 0 < int(tile) <= cleaners.DENOISE_OVERLAP:
            raise ValueError(f"Inkvec Denoise: tile must be 0 (whole image) or larger than "
                             f"the {cleaners.DENOISE_OVERLAP}-px overlap, e.g. 128")
        try:
            restorer = cleaners.load_restorer(device, weights_path)
        except cleaners.CleanerUnavailable as exc:
            raise _cleaner_error("Inkvec Denoise", exc) from exc
        if mode == "auto" and tracer.renderer() is None:
            raise RuntimeError(
                "Inkvec Denoise: mode auto compares the input with its own trace and needs an SVG "
                "renderer. `pip install resvg-py`, or set mode to on."
            )
        rgb, alpha = split_image(image, mask)
        n = rgb.shape[0]
        _, raise_interrupt = _comfy_interrupt()
        pbar = _progress(n)
        images = []
        for i in range(n):
            rgba = _rgba_array(rgb, alpha, i)
            if mode == "auto" and not self._damaged(rgb, alpha, i, rgba):
                images.append(rgb[i])
            else:
                out = cleaners.restore_rgba(restorer, rgba, tile=int(tile))
                images.append(torch.from_numpy(np.ascontiguousarray(out[..., :3])))
            raise_interrupt()
            if pbar is not None:
                pbar.update(1)
        if mask is not None and tuple(mask.shape) == tuple(rgb.shape[:3]):
            out_mask = mask  # passed through as given
        else:
            out_mask = (1.0 - alpha if alpha is not None else torch.zeros(rgb.shape[:3])).contiguous()
        return (torch.stack(images).contiguous(), out_mask)

    @staticmethod
    def _damaged(rgb, alpha, i, rgba) -> bool:
        """`--restore auto`: trace with the CLI defaults, render, measure the interior residual."""
        from . import cleaners

        a8 = _to_uint8(alpha[i]) if alpha is not None else None
        if a8 is not None and int(a8.min()) == 255:
            a8 = None
        try:
            svg = _trace_frame(_to_uint8(rgb[i]), a8, tracer.build_args(), timeout_sec=300)
        except (tracer.InkvecError, binary_manager.BinaryError) as exc:
            raise RuntimeError(f"Inkvec Denoise (auto needs a probe trace): {exc}") from exc
        h, w = rgba.shape[:2]
        model = tracer.render_svg(svg, w, h)
        r = cleaners.interior_residual(rgba, model) if model is not None else None
        if r is None:
            log.info("[inkvec] denoise auto: could not measure the fit; image left unchanged")
            return False
        verdict = "restoring" if r > cleaners.DEGRADED_RESIDUAL else "image left unchanged"
        log.info("[inkvec] denoise auto: interior residual %.3f (threshold %.2f), %s",
                 r, cleaners.DEGRADED_RESIDUAL, verdict)
        return r > cleaners.DEGRADED_RESIDUAL


NODE_CLASS_MAPPINGS = {
    "InkvecTrace": InkvecTrace,
    "InkvecSaveSVG": InkvecSaveSVG,
    "InkvecUpscale": InkvecUpscale,
    "InkvecDenoise": InkvecDenoise,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InkvecTrace": "Inkvec Trace (raster to SVG)",
    "InkvecSaveSVG": "Inkvec Save SVG",
    "InkvecUpscale": "Inkvec Upscale x4 (MambaIRv2)",
    "InkvecDenoise": "Inkvec Denoise (ConvNeXt)",
}
