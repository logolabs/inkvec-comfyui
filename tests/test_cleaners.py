"""The two cleaners on small images. CPU by default; CUDA is exercised too when present.

Each test skips (with the reason) when the optional dependency it needs is missing, so the
trace tests stay runnable on a machine with none of them installed.
"""

import io
import logging

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

COPPER = (201, 117, 74)


def logo(size: int, transparent: bool) -> Image.Image:
    """A flat-colour logo: charcoal ring and copper triangle, on white or on transparency."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0) if transparent else (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    s = size / 64
    d.ellipse([6 * s, 6 * s, 58 * s, 58 * s], outline=(38, 36, 33, 255), width=max(2, int(6 * s)))
    d.polygon([(32 * s, 16 * s), (48 * s, 46 * s), (16 * s, 46 * s)], fill=COPPER + (255,))
    return img


def loadimage(img: Image.Image):
    rgb = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0)[None]
    if "A" in img.getbands():
        mask = (1.0 - torch.from_numpy(np.asarray(img.getchannel("A"), np.float32) / 255.0))[None]
    else:
        mask = torch.zeros((1, 64, 64))
    return rgb, mask


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ============================================================================ upscaler
@pytest.fixture(scope="module")
def sr_node(ink):
    pytest.importorskip("einops", reason="the upscaler needs einops (pip install einops)")
    return ink.nodes.InkvecUpscale()


@pytest.mark.parametrize("device", devices())
def test_upscale_x4_shape_range_determinism_alpha(sr_node, device):
    image, mask = loadimage(logo(32, transparent=True))
    rng_before = torch.random.get_rng_state()
    out1, m1 = sr_node.upscale(image, "4x", device, 256, mask=mask)
    assert torch.equal(rng_before, torch.random.get_rng_state()), "the seeded forward leaked RNG state"
    out2, m2 = sr_node.upscale(image, "4x", device, 256, mask=mask)
    assert tuple(out1.shape) == (1, 128, 128, 3) and tuple(m1.shape) == (1, 128, 128)
    assert out1.dtype == torch.float32 and torch.isfinite(out1).all()
    assert float(out1.min()) >= 0.0 and float(out1.max()) <= 1.0
    assert torch.equal(out1, out2) and torch.equal(m1, m2), "two runs differ despite the pinned seed"
    # Transparent stays transparent (corners), opaque stays opaque (ring and triangle).
    for y, x in [(1, 1), (1, 126), (126, 1), (126, 126)]:
        assert float(m1[0, y, x]) > 0.98, (y, x, float(m1[0, y, x]))
    assert float(m1[0, 76, 64]) < 0.02  # inside the triangle
    # Colour survives: the triangle is still copper.
    assert torch.allclose(out1[0, 76, 64], torch.tensor(COPPER) / 255.0, atol=0.06), out1[0, 76, 64]


def test_upscale_x2_recipe(sr_node):
    image, _ = loadimage(logo(32, transparent=False))
    out, m = sr_node.upscale(image, "2x", "cpu", 256)
    assert tuple(out.shape) == (1, 64, 64, 3) and tuple(m.shape) == (1, 64, 64)
    assert float(m.max()) < 0.01
    # Flats recoloured onto the source: the white ground is white.
    assert float(out[0, 2, 2].min()) > 0.97


def test_upscale_tiles_blend(ink, sr_node):
    """Tiles smaller than the image give the same size and no gross seam. Four tiles on CPU
    take minutes, so this runs on CUDA when there is one."""
    device = devices()[-1]
    image, _ = loadimage(logo(48, transparent=False))
    whole, _ = sr_node.upscale(image, "4x", device, 256)
    tiled, _ = sr_node.upscale(image, "4x", device, 40)  # 40-px tiles, 32 overlap: 2x2 tiles
    assert ink.cleaners.sr_tile_count(48, 48, 40) == 4
    assert tiled.shape == whole.shape
    assert (tiled - whole).abs().mean().item() < 0.02


# ============================================================================ denoiser
@pytest.fixture(scope="module")
def dn(ink):
    pytest.importorskip("onnxruntime", reason="the denoiser needs onnxruntime (pip install onnxruntime)")
    return ink.nodes.InkvecDenoise()


def _reference_restore(sess, rgb01: np.ndarray) -> np.ndarray:
    """The Python reference from the Logolabs/inkvec-denoiser-001 model card, verbatim."""
    MULTIPLE = 16
    SNAP_LEVELS = 6 / 255.0
    h, w, _ = rgb01.shape
    ph, pw = -(-h // MULTIPLE) * MULTIPLE, -(-w // MULTIPLE) * MULTIPLE
    padded = np.pad(rgb01, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    chw = padded.transpose(2, 0, 1)[None].astype(np.float32)
    (out,) = sess.run(["restored"], {"image": chw})
    out = out[0].transpose(1, 2, 0)[:h, :w]
    out = np.round(np.clip(out, 0, 1) * 255) / 255
    lo, hi = out <= SNAP_LEVELS, out >= 1 - SNAP_LEVELS
    snap_hi = hi.all(axis=-1, keepdims=True)
    snap_lo = lo.all(axis=-1, keepdims=True)
    out = np.where(snap_hi, 1.0, np.where(snap_lo, 0.0, out))
    return out.astype(np.float32)


@pytest.mark.parametrize("device", devices())
def test_denoise_on_matches_the_reference(ink, dn, device):
    damaged = jpeg(logo(64, transparent=False), 30)
    image, _ = loadimage(damaged)
    out, m = dn.denoise(image, "on", device, 0)
    assert tuple(out.shape) == (1, 64, 64, 3) and tuple(m.shape) == (1, 64, 64)
    arr = out[0].numpy()
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert np.allclose(arr * 255.0, np.round(arr * 255.0), atol=1e-3), "not quantised to 8-bit levels"
    r = ink.cleaners.load_restorer(device)
    ref = _reference_restore(r.session, image[0].numpy())
    diff = np.abs(arr - ref)
    assert diff.max() <= 1.0 / 255.0 + 1e-6 and (diff > 0).mean() < 1e-3, (diff.max(), (diff > 0).mean())
    out2, _ = dn.denoise(image, "on", device, 0)
    assert torch.equal(out, out2)


def test_denoise_carries_alpha_and_composites_on_white(dn):
    image, mask = loadimage(logo(64, transparent=True))
    out, m = dn.denoise(image, "on", "cpu", 0, mask=mask)
    assert torch.equal(m, mask), "the mask must pass through unchanged"
    # Transparent pixels were composited onto white before restoring, as the engine does.
    assert torch.equal(out[0, 1, 1], torch.ones(3))


def test_denoise_tiled(dn):
    image, _ = loadimage(jpeg(logo(96, transparent=False), 30))
    whole, _ = dn.denoise(image, "on", "cpu", 0)
    tiled, _ = dn.denoise(image, "on", "cpu", 64)  # 64-px tiles, 32 overlap: 2x2 tiles
    assert tiled.shape == whole.shape == (1, 96, 96, 3)
    assert (tiled - whole).abs().mean().item() < 0.01
    with pytest.raises(ValueError):
        dn.denoise(image, "on", "cpu", 32)  # not larger than the overlap


def test_snap_extremes_matches_the_engine_test(ink):
    # crates/inkvec-restore/src/lib.rs: snap_extremes_only_touches_pixels_near_an_extreme
    rgb = np.array([[1.0, 1.0, 1.0], [0.98, 0.99, 1.0], [0.5, 0.5, 0.5], [0.02, 0.0, 0.01], [0.5, 0.0, 0.0]],
                   np.float32)
    out = ink.cleaners.snap_extremes(rgb)
    assert out[1].tolist() == [1.0, 1.0, 1.0] and out[2].tolist() == [0.5, 0.5, 0.5]
    assert out[3].tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(out[4], [0.5, 0.0, 0.0])


def test_interior_residual_scale_matches_the_engine(ink):
    # crates/inkvec-sr/src/detect.rs: a uniform 4-level offset reads 4/sqrt(3).
    a = np.full((32, 32, 4), 0.5, np.float32)
    a[..., 3] = 1.0
    b = a.copy()
    b[..., :3] += 4.0 / 255.0
    r = ink.cleaners.interior_residual(a, b)
    assert abs(r - 4.0 / np.sqrt(3.0)) < 0.05, r
    assert ink.cleaners.interior_residual(a, a) < 1e-9


def test_denoise_auto_decides_like_the_cli(ink, dn, caplog):
    pytest.importorskip("resvg_py", reason="auto mode renders the probe trace with resvg-py")
    clean = logo(96, transparent=False)
    image, _ = loadimage(clean)
    with caplog.at_level(logging.INFO, logger="inkvec-comfyui"):
        out, _ = dn.denoise(image, "auto", "cpu", 0)
    assert torch.equal(out, image), "clean input must pass through unchanged"
    damaged, _ = loadimage(jpeg(clean, 20))
    with caplog.at_level(logging.INFO, logger="inkvec-comfyui"):
        out, _ = dn.denoise(damaged, "auto", "cpu", 0)
    assert not torch.equal(out, damaged), "JPEG q20 input should be restored"
    notes = [r.getMessage() for r in caplog.records if "denoise auto" in r.getMessage()]
    print("\n  " + "\n  ".join(notes))
    assert len(notes) == 2 and "left unchanged" in notes[0] and "restoring" in notes[1]


def test_missing_dependency_message(ink, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    image, _ = loadimage(logo(32, transparent=False))
    with pytest.raises(RuntimeError, match="pip install onnxruntime"):
        ink.nodes.InkvecDenoise().denoise(image, "on", "cpu", 0)
